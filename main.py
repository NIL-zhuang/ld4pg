import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import loggers as pl_logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, DeviceStatsMonitor
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoConfig, AutoTokenizer, BartForConditionalGeneration, BartTokenizer

from ld4pg.data.data_module import get_dataset, DataModule
from ld4pg.model.denoising_diffusion import GaussianDiffusion
from ld4pg.model.diffusion_transformer import DiffusionTransformer

FAST_DEV_RUN = True
CPU_TEST = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/config.yaml", help="path to config which construct model")
    parser.add_argument("--ckpt", type=str, default=None, help="path to model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible results)")
    parser.add_argument("-n", "--name", type=str, default="", help="postfix for dir")
    parser.add_argument(
        "-m", "--mode", type=str, default='eval',
        choices=['train', 'eval', 'resume', 'interact'],
        help="train, resume or eval"
    )

    args = parser.parse_args()
    return args


def build_dataset(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.tokenizer)
    dataset = get_dataset(cfg.name)
    dataset_module = DataModule(
        cfg=cfg.params,
        tokenizer=tokenizer,
        train_dataset=dataset[0],
        valid_dataset=dataset[1],
        test_dataset=dataset[2],
        inf_train_dataloader=False,
    )
    return dataset_module


def build_denoise_network(cfg: DictConfig):
    config = AutoConfig.from_pretrained(cfg.diffusion.params.enc_dec_model)
    transformer_cfg = cfg.transformer.params
    assert transformer_cfg.latent_dim % transformer_cfg.attention_head_dim == 0, f'Transformer dimension must be divisible by ATTN_HEAD_DIM'
    assert transformer_cfg.latent_dim == config.d_model, "Transformer latent should be same dimension as encoder latent space"

    model = DiffusionTransformer(
        tx_dim=transformer_cfg.latent_dim,
        tx_depth=transformer_cfg.depth,
        heads=transformer_cfg.latent_dim // transformer_cfg.attention_head_dim,
        latent_dim=transformer_cfg.latent_dim,
        max_seq_len=cfg.params.max_seq_len,
        self_condition=cfg.params.self_condition,
        scale_shift=cfg.scale_shift,
        dropout=transformer_cfg.dropout,
        conditional=True,
        unconditional_prob=cfg.params.unconditional_prob,
    )
    return model


def build_model(cfg: DictConfig):
    model = build_denoise_network(cfg)
    diffusion_cfg = cfg.diffusion.params
    encoder = BartForConditionalGeneration.from_pretrained(diffusion_cfg.enc_dec_model).get_encoder()
    diffusion = GaussianDiffusion(
        model,
        encoder,
        cfg=cfg.diffusion.params,
        max_seq_len=cfg.params.max_seq_len,
        timesteps=diffusion_cfg.timesteps,  # number of steps
        sampling_timesteps=diffusion_cfg.sampling_timesteps,  # number of sampling timesteps (ddim for faster inference)
        loss_type=diffusion_cfg.loss_type,  # L1 or L2
        beta_schedule=diffusion_cfg.beta_schedule,
        p2_loss_weight_gamma=diffusion_cfg.p2_loss_weight_gamma,
        objective=diffusion_cfg.objective,
        ddim_sampling_eta=diffusion_cfg.ddim_sampling_eta,
    )
    return diffusion


def load_model(cfg: DictConfig, ckpt: str):
    model = build_denoise_network(cfg)
    diffusion_cfg = cfg.diffusion.params
    encoder = BartForConditionalGeneration.from_pretrained(diffusion_cfg.enc_dec_model).get_encoder()
    diffusion = GaussianDiffusion.load_from_checkpoint(
        ckpt,
        model=model,
        encoder=encoder
    )
    return diffusion


def get_save_path(output_dir: str, dataset_name: str, model_name: str):
    local_rank = os.environ.get('LOCAL_RANK', 0)
    if local_rank == 0:
        output_dir = os.path.join(
            output_dir,
            f"{dataset_name}_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M')}"
        )
        os.environ['RUN_OUTPUT_DIR'] = output_dir
    else:
        output_dir = os.environ['RUN_OUTPUT_DIR']
    return output_dir


def build_trainer(cfg, save_path="saved_models"):
    callbacks = [
        ModelCheckpoint(
            dirpath=save_path,
            filename='{step}-{val/loss_ema:.2f}',
            every_n_train_steps=5000,
        ),
        RichProgressBar(refresh_rate=1),
        DeviceStatsMonitor(),
    ]
    logger = pl_logger.TensorBoardLogger(save_dir=save_path)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_steps=cfg.num_train_steps,
        accelerator='gpu' if torch.cuda.is_available() and not CPU_TEST else 'cpu',
        devices=-1 if torch.cuda.is_available() and not CPU_TEST else 1,
        precision=cfg.precision,
        auto_select_gpus=True,
        log_every_n_steps=20,
        strategy=DDPStrategy(find_unused_parameters=True),
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        fast_dev_run=FAST_DEV_RUN
    )
    return trainer


def sample(
        model: pl.LightningModule,
        trainer: pl.Trainer,
        dataset: DataLoader,
        cfg: DictConfig
):
    enc_dec_model = BartForConditionalGeneration.from_pretrained(cfg.model.diffusion.params.enc_dec_model)
    tokenizer = BartTokenizer.from_pretrained(cfg.data.params.tokenizer)

    result = []
    encoder_output_list = trainer.predict(model, dataset)
    for encoder_output, mask in encoder_output_list:
        samples = enc_dec_model.generate(
            encoder_outputs=encoder_output,
            attention_mask=mask.clone(),
            **cfg.model.sample.beam
        )
        text_list = [
            tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in samples
        ]
        text_list = [
            text.strip() for text in text_list
            if len(text.strip()) > 0
        ]
        result += text_list
    return result


def main(opt: argparse.Namespace) -> None:
    pl.seed_everything(opt.seed)
    cfg: DictConfig = OmegaConf.load(f"{opt.config}")

    save_path = get_save_path(cfg.train.output_dir, cfg.data.name, opt.name)
    trainer = build_trainer(cfg.train.params, save_path)

    dataset = build_dataset(cfg.data)

    if opt.mode == 'train':
        model = build_model(cfg.model)
        trainer.fit(model, dataset)
    elif opt.mode == 'eval':
        model = load_model(cfg.model, opt.ckpt)
        model.eval()
        model.freeze()
        result = sample(model, trainer, dataset.test_dataloader(), cfg)
        print('\n'.join(result))
    elif opt.mode == 'resume':
        model = load_model(cfg.model, opt.ckpt)
        trainer.fit(model, dataset)
    elif opt.mode == 'interact':
        model = load_model(cfg.model, opt.ckpt)
        model.eval()
        model.freeze()


if __name__ == '__main__':
    option = parse_args()
    main(option)

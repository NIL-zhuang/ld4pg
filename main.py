import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import loggers as pl_logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, DeviceStatsMonitor, RichModelSummary
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoConfig, AutoTokenizer

from ld4pg.dataset.data_module import get_dataset, DataModule
from ld4pg.model.denoising_diffusion import GaussianDiffusion
from ld4pg.model.diffusion_transformer import DiffusionTransformer

FAST_DEV_RUN = False


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        elif v.lower() in ('yes', 'true', 'y', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'n', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/config.yaml", help="path to config which construct model")
    parser.add_argument("--ckpt", type=str, default=None, help="path to model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible results)")
    parser.add_argument("-n", "--name", type=str, default="", help="postfix for dir")

    parser.add_argument('-t', "--train", type=str2bool, const=True, default=True, nargs='?', help="train model")
    parser.add_argument('-r', "--resume", type=str2bool, const=True, default=False, nargs='?', help="resume training")
    parser.add_argument('-e', "--eval", type=str2bool, const=True, default=False, nargs='?', help="evaluate model")

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


def build_model(cfg: DictConfig):
    config = AutoConfig.from_pretrained(cfg.enc_dec_model)
    assert cfg.tx_dim % ATTN_HEAD_DIM == 0, f'Transformer dimension must be divisible by {ATTN_HEAD_DIM}'
    model = DiffusionTransformer(
        tx_dim=cfg.tx_dim,
        tx_depth=cfg.tx_depth,
        heads=cfg.tx_dim // ATTN_HEAD_DIM,
        latent_dim=config.d_model,
        max_seq_len=cfg.max_token_len,
        self_condition=cfg.self_condition,
        scale_shift=cfg.scale_shift,
        dropout=0 if cfg.disable_dropout else 0.1,
        conditional=True,
        unconditional_prob=cfg.unconditional_prob,
    )
    diffusion = GaussianDiffusion(
        model,
        cfg,
        max_seq_len=model.max_seq_len,
        timesteps=cfg.timesteps,  # number of steps
        sampling_timesteps=cfg.sampling_timesteps,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type=cfg.loss_type,  # L1 or L2
        beta_schedule=cfg.beta_schedule,
        p2_loss_weight_gamma=cfg.p2_loss_weight_gamma,
        objective=cfg.objective,
        ddim_sampling_eta=cfg.ddim_sampling_eta,
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
            filename='{step}-val_loss_ema{val/loss_ema:.2f}',
            every_n_train_steps=5000,
        ),
        RichProgressBar(refresh_rate=1),
        DeviceStatsMonitor(),
        RichModelSummary()
    ]
    logger = pl_logger.TensorBoardLogger(save_dir=save_path)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_steps=cfg.num_train_steps,
        accelerator='gpu' if torch.cuda.is_available() and not FAST_DEV_RUN else 'cpu',
        devices=-1 if torch.cuda.is_available() and not FAST_DEV_RUN else 1,
        precision=cfg.precision,
        auto_select_gpus=True,
        log_every_n_steps=20,
        strategy=DDPStrategy(find_unused_parameters=True),
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        fast_dev_run=FAST_DEV_RUN
    )
    return trainer


def main(opt: argparse.Namespace) -> None:
    pl.seed_everything(opt.seed)
    cfg: DictConfig = OmegaConf.load(f"{opt.config}")

    save_path = get_save_path(cfg.train.output_dir, cfg.data.name, opt.name)
    trainer = build_trainer(cfg.train.params, save_path)

    dataset = build_dataset(cfg.data)

    if opt.train:
        model = build_model(cfg.model)
        trainer.fit(model, dataset)
    elif opt.eval:
        model = GaussianDiffusion.load_from_checkpoint(opt.ckpt)
        trainer.test(model, dataset.test_dataloader())
    elif opt.resume:
        model = GaussianDiffusion.load_from_checkpoint(opt.ckpt)
        trainer.fit(model, dataset)


if __name__ == '__main__':
    option = parse_args()
    main(option)

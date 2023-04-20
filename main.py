import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import loggers as pl_logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizer

from ld4pg.data.data_module import get_dataset, DataModule
from ld4pg.models.diffusion.ddpm import LatentDiffusion

FAST_DEV_RUN = False
CPU_TEST = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/config.yaml", help="path to config which construct model")
    parser.add_argument("--ckpt", type=str, default=None, help="path to model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible results)")
    parser.add_argument("-n", "--name", type=str, default="", help="postfix for dir")
    parser.add_argument("--tgt", type=str, default="result.txt", help="target file path")
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


def build_model(cfg: DictConfig):
    diffusion_cfg = cfg.diffusion.params
    first_stage_model = BartForConditionalGeneration.from_pretrained(diffusion_cfg.enc_dec_model)
    first_stage_tokenizer = BartTokenizer.from_pretrained(diffusion_cfg.enc_dec_model)
    model = LatentDiffusion(
        model_cfg=cfg.transformer,
        first_stage_model=first_stage_model,
        cond_stage_model=first_stage_model,
        first_stage_tokenizer=first_stage_tokenizer,
        condition_key=diffusion_cfg.condition_key,
        beta_schedule=diffusion_cfg.beta_schedule,
        parameterization=diffusion_cfg.parameterization,
        loss_type=diffusion_cfg.loss_type,
        timesteps=diffusion_cfg.timesteps,
        max_seqlen=cfg.params.max_seq_len,
        use_ema=diffusion_cfg.use_ema,
        scale_factor=diffusion_cfg.scale_factor,
        scale_mean=diffusion_cfg.scale_mean,
        unconditional_prob=diffusion_cfg.unconditional_prob,
        learning_rate=diffusion_cfg.learning_rate,
        normalize=diffusion_cfg.normalize,
        learn_logvar=diffusion_cfg.learn_logvar,
        sample_strategy=cfg.sample.beam
    )
    return model


def load_model(cfg: DictConfig, ckpt: str):
    diffusion_cfg = cfg.diffusion.params
    first_stage_model = BartForConditionalGeneration.from_pretrained(diffusion_cfg.enc_dec_model)
    first_stage_tokenizer = BartTokenizer.from_pretrained(diffusion_cfg.enc_dec_model)
    model = LatentDiffusion.load_from_checkpoint(
        ckpt,
        first_stage_model=first_stage_model,
        cond_stage_model=first_stage_model,
        first_stage_tokenizer=first_stage_tokenizer,
    )
    return model


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
            monitor='val/loss_ema',
            filename='step{step}-valema{val/loss_ema:.2f}',
            every_n_train_steps=10000,
            auto_insert_metric_name=False,
            save_top_k=-1,
            save_on_train_epoch_end=True,
        ),
        RichProgressBar(refresh_rate=1),
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
        strategy=DDPStrategy(),
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        fast_dev_run=FAST_DEV_RUN
    )
    return trainer


def build_eval_trainer():
    callbacks = [RichProgressBar(refresh_rate=1)]
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() and not CPU_TEST else 'cpu',
        callbacks=callbacks,
        logger=False,
        fast_dev_run=FAST_DEV_RUN
    )
    return trainer


def main(opt: argparse.Namespace) -> None:
    pl.seed_everything(opt.seed)
    cfg: DictConfig = OmegaConf.load(f"{opt.config}")

    if opt.mode == 'train':
        save_path = get_save_path(cfg.train.output_dir, cfg.data.name, opt.name)
    else:
        save_path = os.sep.join(opt.ckpt.split('/')[:2])
    print(f"Model save path: {save_path}")

    if opt.mode in ('train', 'resume'):
        trainer = build_trainer(cfg.train.params, save_path)
    else:
        trainer = build_eval_trainer()

    dataset = build_dataset(cfg.data)

    if opt.mode == 'train':
        model = build_model(cfg.model)
        trainer.fit(model, dataset)
    elif opt.mode == 'resume':
        model = load_model(cfg.model, opt.ckpt)
        trainer.fit(model, dataset)
    elif opt.mode == 'eval':
        model = load_model(cfg.model, opt.ckpt)
        model.eval()
        model.freeze()
        results = trainer.predict(model, dataset.test_dataloader())
        texts = [text for result in results for text in result]
        # print('\n'.join(texts))
        with open(opt.tgt, 'w+', encoding='utf-8') as f:
            f.write('\n'.join(texts))
    elif opt.mode == 'interact':
        raise NotImplementedError("Interact mode is not implemented yet")


if __name__ == '__main__':
    option = parse_args()
    main(option)

import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from transformers import AutoConfig, AutoTokenizer

from ld4pg.config import *
from ld4pg.data.data_module import get_dataset, DataModule
from ld4pg.model.denoising_diffusion import GaussianDiffusion
from ld4pg.model.diffusion_transformer import DiffusionTransformer
from ld4pg.utils.arguments import get_parser

ATTN_HEAD_DIM = DEFAULT_DIM_HEAD
FAST_DEV_RUN = False


def build_model(args):
    config = AutoConfig.from_pretrained(args.enc_dec_model)
    assert args.tx_dim % ATTN_HEAD_DIM == 0, f'Transformer dimension must be divisible by {ATTN_HEAD_DIM}'
    model = DiffusionTransformer(
        tx_dim=args.tx_dim,
        tx_depth=args.tx_depth,
        heads=args.tx_dim // ATTN_HEAD_DIM,
        latent_dim=config.d_model,
        max_seq_len=args.max_token_len,
        self_condition=args.self_condition,
        scale_shift=args.scale_shift,
        dropout=0 if args.disable_dropout else 0.1,
        conditional=True,
        unconditional_prob=args.unconditional_prob,
    )
    diffusion = GaussianDiffusion(
        model,
        args,
        max_seq_len=model.max_seq_len,
        timesteps=args.timesteps,  # number of steps
        sampling_timesteps=args.sampling_timesteps,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type=args.loss_type,  # L1 or L2
        beta_schedule=args.beta_schedule,
        p2_loss_weight_gamma=args.p2_loss_weight_gamma,
        objective=args.objective,
        ddim_sampling_eta=args.ddim_sampling_eta,
    )
    return diffusion


def build_dataset(args):
    tokenizer = AutoTokenizer.from_pretrained(args.enc_dec_model)
    dataset = get_dataset(args.dataset_name)
    dataset_module = DataModule(
        cfg=args,
        tokenizer=tokenizer,
        train_dataset=dataset[0],
        valid_dataset=dataset[1],
        test_dataset=dataset[2],
        inf_train_dataloader=False,
    )
    return dataset_module


MODEL_NAME = "ld4pg-normalize"


def get_save_path(args):
    local_rank = os.environ.get('LOCAL_RANK', 0)
    if local_rank == 0:
        output_dir = os.path.join(args.output_dir, f"{args.dataset_name}_{MODEL_NAME}_{datetime.now().strftime('%Y%m%d-%H%M')}")
        os.environ['RUN_OUTPUT_DIR'] = output_dir
    else:
        output_dir = os.environ['RUN_OUTPUT_DIR']
    return output_dir


def build_trainer(args):
    save_path = get_save_path(args)
    callbacks = [
        ModelCheckpoint(
            dirpath=save_path,
            filename='{step}-val_loss_ema{val/loss_ema:.2f}',
            every_n_train_steps=5000,
        ),
        RichProgressBar(refresh_rate=1),
    ]
    logger = pl_logger.TensorBoardLogger(save_dir=save_path)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_steps=args.num_train_steps,
        accelerator='gpu' if torch.cuda.is_available() and not FAST_DEV_RUN else 'cpu',
        devices=-1 if torch.cuda.is_available() and not FAST_DEV_RUN else 1,
        precision=32,
        auto_select_gpus=True,
        log_every_n_steps=20,
        strategy=DDPStrategy(find_unused_parameters=True),
        accumulate_grad_batches=args.accumulate_grad_batches,
        fast_dev_run=FAST_DEV_RUN
        # overfit_batches=10
    )
    return trainer


def train(args):
    pl.seed_everything(args.seed)
    diffusion = build_model(args)
    dataset = build_dataset(args)
    trainer = build_trainer(args)
    trainer.fit(diffusion, dataset)


def main():
    parser = get_parser()
    args = parser.parse_args()

    args.normalize_latent = True

    print(args)
    train(args)


if __name__ == '__main__':
    main()

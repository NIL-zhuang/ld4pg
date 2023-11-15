import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pympler import asizeof
from pytorch_lightning import loggers as pl_logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.strategies import DDPStrategy
from transformers import BartForConditionalGeneration, BartTokenizerFast as BartTokenizer

from ld4pg.config import *
from ld4pg.data import get_dataset
from ld4pg.data.controlnet_data_module import ControlNetKeywordDataModule
from ld4pg.models.control_net.controlnet import ControlNetModel
from ld4pg.models.control_net.controlnet_pipeline import LDPControlNetPipeline
from ld4pg.models.diffusion.ddpm import LatentDiffusion
from ld4pg.util import arg_transform

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FAST_DEV_RUN = False
CPU_TEST = False


def get_save_path(output_dir: str, dataset_name: str, model_name: str):
    local_rank = os.environ.get('LOCAL_RANK', 0)
    if local_rank == 0:
        output_dir = os.path.join(
            output_dir,
            f"{dataset_name}_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M')}"
        )
        os.environ['RUN_OUTPUT_DIR'] = output_dir
        if FAST_DEV_RUN is False:
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.environ['RUN_OUTPUT_DIR']
    return output_dir


def build_dataset(cfg: DictConfig):
    tokenizer = BartTokenizer.from_pretrained(cfg.params.tokenizer)
    dataset = get_dataset(cfg.name)
    dataset_module = ControlNetKeywordDataModule(
        cfg=cfg.params,
        tokenizer=tokenizer,
        data_path=os.path.join(DATASET_PATH, cfg.name),
        train_dataset=dataset[0][:1000] if FAST_DEV_RUN else dataset[0],
        # train_dataset=dataset[0],
        valid_dataset=dataset[1],
        test_dataset=dataset[2],
        inf_train_dataloader=False,
    )
    return dataset_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/ldp/config_chatgpt.yaml", help="conf file")
    parser.add_argument("--ckpt", type=str, default=None, help="backbone ckpt")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("-n", "--name", type=str, default="", help="dir postfix")
    parser.add_argument("-u", "--update", nargs='+', default=[], help='update parameters')
    parser.add_argument("--save_path", type=str, default="saved_models", help="path to save model")
    parser.add_argument(
        "-m", "--mode", type=str, default='train',
        choices=['train', 'resume'], help="train or resume"
    )

    args = parser.parse_args()
    return args


def build_trainer(cfg, save_path="saved_models"):
    callbacks = [
        ModelCheckpoint(
            dirpath=save_path,
            monitor='val/loss_ema',
            filename='step{step}-valema{val/loss_ema:.2f}',
            every_n_train_steps=5000,
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
        fast_dev_run=FAST_DEV_RUN,
    )
    return trainer


def build_controlnet_train_pipeline(config: DictConfig, ckpt_path: str):
    diffusion_cfg = config.diffusion.params
    controlnet_cfg = config.controlnet.params
    first_stage_model = BartForConditionalGeneration.from_pretrained(diffusion_cfg.enc_dec_model)
    cond_stage_model = first_stage_model
    first_stage_tokenizer = BartTokenizer.from_pretrained(diffusion_cfg.enc_dec_model)
    control_cond_stage_model = BartForConditionalGeneration.from_pretrained(controlnet_cfg.cn_model)

    def build_controlnet_model(ckpt: str):
        cn_input_key = controlnet_cfg.additional_input_key
        cn_input_mask = controlnet_cfg.additional_input_mask
        model = ControlNetModel.load_from_checkpoint(
            ckpt,
            first_stage_model=first_stage_model,
            cond_stage_model=first_stage_model,
            first_stage_tokenizer=first_stage_tokenizer,
            additional_input_key=cn_input_key,
            additional_input_mask=cn_input_mask,
            strict=False,
        )
        return model

    def load_ldp_model(ckpt: str):
        model = LatentDiffusion.load_from_checkpoint(
            ckpt,
            first_stage_model=first_stage_model,
            cond_stage_model=first_stage_model,
            first_stage_tokenizer=first_stage_tokenizer
        )
        model.eval()
        model.freeze()
        return model

    ldp = load_ldp_model(ckpt_path)
    controlnet = build_controlnet_model(ckpt_path)
    ldp_controlnet_pipeline = LDPControlNetPipeline(
        diffusion_cfg, text_cond_stage_model=cond_stage_model,
        first_stage_model=first_stage_model, tokenizer=first_stage_tokenizer,
        ldp=ldp, controlnet=controlnet,
        learning_rate=config.controlnet.params.learning_rate,
        controlnet_cond_stage_model=control_cond_stage_model.get_encoder(),
        cn_scale_factor=controlnet_cfg.scale_factor,
        cn_scale_mean=controlnet_cfg.scale_mean

    )
    return ldp_controlnet_pipeline


def main(opt: argparse.Namespace):
    cfg: DictConfig = OmegaConf.load(f"{opt.config}")
    for param in opt.update:
        k, v = param.split("=")
        OmegaConf.update(cfg, k, arg_transform(v), merge=True)

    save_dir = get_save_path(
        cfg.train.output_dir if opt.save_path is None else opt.save_path,
        cfg.data.name,
        opt.name
    )
    print(f"Model save path: {save_dir}")

    dataset = build_dataset(cfg.data)
    print(f"Size of dataset is {asizeof.asizeof(dataset) / 1024 / 1024} GB")
    pipeline = build_controlnet_train_pipeline(cfg.model, opt.ckpt).cuda()
    print(f"Size of controlnet train pipeline is {asizeof.asizeof(pipeline) / 1024 / 1024} GB")
    trainer = build_trainer(cfg.train.params, save_dir)
    print(f"Size of controlnet trainer is {asizeof.asizeof(trainer) / 1024 / 1024} GB")

    if os.environ.get('LOCAL_RANK', 0) == 0 and FAST_DEV_RUN is False:
        OmegaConf.save(config=cfg, f=os.path.join(save_dir, 'conf.yaml'))
    trainer.fit(pipeline, dataset)


if __name__ == '__main__':
    option = parse_args()
    pl.seed_everything(option.seed)
    main(option)

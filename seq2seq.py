import argparse
import os
from glob import glob

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
from rich.progress import track
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizer

from ld4pg.data import get_dataset
from ld4pg.data.data_module import DataModule
from ld4pg.models.diffusion.ddpm import LatentDiffusion
from ld4pg.util import arg_transform
from ld4pg.config import SAMPLE_STRATEGY

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/config_qqp.yaml", help="config to construct model")
    parser.add_argument("--ckpt", type=str, default=None, help="path to model checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="path to model checkpoint save dir")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible results)")
    parser.add_argument("--tgt", type=str, default="/home/zhuangzy/result", help="target file dir")
    parser.add_argument("--fname", type=str, default=None, help="target file name")
    parser.add_argument("-u", "--update", nargs='+', default=[], help='update parameters')
    parser.add_argument("--sampler", type=str, default='dpm', choices=['ddim', 'dpm'], help="sampler type")
    parser.add_argument("--steps", type=int, default=25, help="number of steps")
    args = parser.parse_args()
    return args


def build_dataset(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.tokenizer)
    dataset = get_dataset(cfg.name)
    dataset_module = DataModule(
        cfg=cfg.params,
        tokenizer=tokenizer,
        test_dataset=dataset[2],
        inf_train_dataloader=False,
    )
    return dataset_module


def load_model(enc_dec_model: str, ckpt: str):
    first_stage_model = BartForConditionalGeneration.from_pretrained(enc_dec_model)
    first_stage_tokenizer = BartTokenizer.from_pretrained(enc_dec_model)
    model = LatentDiffusion.load_from_checkpoint(
        ckpt,
        first_stage_model=first_stage_model,
        cond_stage_model=first_stage_model,
        first_stage_tokenizer=first_stage_tokenizer,
    )
    model.eval()
    model.freeze()
    return model


def predict(model, data_loader, steps: int = 25, sampler='dpm'):
    results = []
    with torch.no_grad(), model.ema_scope():
        for batch in track(data_loader, description="Generating..."):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            x, x_mask, c, c_mask = model.get_input(batch)
            texts = model.generate_text(
                c, c_mask, x_mask, batch_size=c.shape[0],
                sample_strategy=SAMPLE_STRATEGY['nucleus'],
                verbose=False, sampler=sampler, steps=steps,
            )
            results += texts
    return results


def main(opt: argparse.Namespace):
    pl.seed_everything(opt.seed)
    cfg: DictConfig = OmegaConf.load(opt.config)
    for param in opt.update:
        k, v = param.split("=")
        OmegaConf.update(cfg, k, arg_transform(v), merge=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = build_dataset(cfg.data)

    if opt.ckpt is not None:
        ckpt_list = [opt.ckpt]
    elif opt.ckpt_dir is not None:
        ckpt_list = glob(f"{opt.ckpt_dir}/*.ckpt")
        # os.makedirs(os.path.join("results", os.path.split(opt.ckpt_dir)[1]), exist_ok=True)
    else:
        raise ValueError("You must config either ckpt or ckpt path")

    for m_path in tqdm(sorted(ckpt_list, reverse=True), desc="Evaluating models..."):
        # get model step, e.g. "step10000-val_ema123.45.ckpt" -> "step10000"
        m_name = os.path.splitext(os.path.split(m_path)[-1])[0].split('-')[0]
        if opt.fname is not None:
            m_name += f"-{opt.fname}"
        print(f"generating {m_name}...")

        model: LatentDiffusion = load_model(cfg.model.diffusion.params.enc_dec_model, m_path).to(device)

        results = predict(model, dataset.test_dataloader(), steps=opt.steps, sampler=opt.sampler)
        with open(os.path.join(opt.tgt, f"{m_name}.txt"), 'w+', encoding='utf-8') as f:
            f.write('\n'.join(results))


if __name__ == '__main__':
    options = parse_args()
    main(options)

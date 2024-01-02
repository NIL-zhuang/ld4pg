import argparse
import os
from collections import defaultdict
from typing import List

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from transformers import BartTokenizer, BartForConditionalGeneration

from ld4pg.config import *
from ld4pg.data import get_dataset
from ld4pg.data.data_module import DataModule
from ld4pg.models.diffusion.ddpm import LatentDiffusion

interp_method = "linear_mean"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default="/home/data_91_d/zhuangzy/latent_diffusion/saved_models/qqp_ldp/conf.yaml",
        help="path to config which construct model")
    parser.add_argument(
        "--ckpt", type=str,
        default="/home/data_91_d/zhuangzy/latent_diffusion/saved_models/qqp_ldp/step210000-valema129.11.ckpt",
        required=False, help="path to model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="the seed for sampling")
    parser.add_argument("--save_path", type=str, default="/home/zhuangzy/interpolation")
    args = parser.parse_args()
    return args


def build_dataset(cfg: DictConfig):
    tokenizer = BartTokenizer.from_pretrained(cfg.params.tokenizer)
    dataset = get_dataset(cfg.name)
    dataset_module = DataModule(
        cfg=cfg.params,
        tokenizer=tokenizer,
        test_dataset=dataset[2],
        inf_train_dataloader=False,
    )
    return dataset_module


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
    model.eval()
    model.freeze()
    return model


def build_input(model: LatentDiffusion, src: List[str], cond: List[str]):
    tokenizer = model.first_stage_tokenizer

    def tokenize(sent: List[str]):
        return tokenizer(
            sent, return_tensors="pt", padding="max_length", truncation=True,
            max_length=64, return_attention_mask=True, add_special_tokens=True
        )

    src_token = tokenize(src)
    cond_token = tokenize(cond)
    latent, mask = model.get_first_stage_encoding(src_token['input_ids'].cuda(), src_token['attention_mask'].cuda())
    cond, cond_mask = model.get_conditioning(cond_token['input_ids'].cuda(), cond_token['attention_mask'].cuda())
    return latent, mask, cond, cond_mask


def build_linear_interp(interp, mask):
    return interp, mask


def interpolation(
        sources: List[str],
        conditions: List[str],
        interps: List[str],
        model: LatentDiffusion,
        alpha: float = 0.5,
        noisy_time: float = 0.5
):
    assert noisy_time <= 1.0, f"noisy proportion should <= 1.0, current is {noisy_time}"
    timestep = torch.tensor(int(noisy_time * model.num_timesteps), dtype=torch.int64).repeat(len(interps))

    model = model.cpu()
    latent, latent_mask, cond, cond_mask = build_input(model, sources, conditions)
    interp_latent, interp_latent_mask, _, _ = build_input(model, interps, conditions)

    if interp_method == 'linear':
        pass
    elif interp_method == 'linear_mean':
        interp_latent = torch.sum(interp_latent * interp_latent_mask.unsqueeze(-1), dim=1) / \
                        torch.sum(interp_latent_mask, dim=1).unsqueeze(-1)
        interp_latent = torch.unsqueeze(interp_latent, 1)
    elif interp_method == "":
        raise NotImplementedError("on going")

    interp_latent = alpha * latent + (1 - alpha) * interp_latent
    x_start = model.q_sample(interp_latent, timestep)

    if torch.cuda.is_available():
        model = model.cuda()
        cond = cond.cuda()
        cond_mask = cond_mask.cuda()
        x_start = x_start.cuda()

    sample, intermediates, latent_mask = model.sample_log(
        cond, cond_mask, cond_mask, batch_size=len(interps), sampler="dpm", steps=25,
        log_every_t=200, verbose=False, x_T=x_start, t_start=noisy_time,
        return_intermediate=True,
    )
    text = model.decode_first_stage(sample, latent_mask)
    print("\n".join(text))


def generate(
        sources: List[str],
        conditions: List[str],
        model: LatentDiffusion,
):
    latent, latent_mask, cond, cond_mask = build_input(model, sources, conditions)

    if torch.cuda.is_available():
        model = model.cuda()
        cond = cond.cuda()
        cond_mask = cond_mask.cuda()
        latent_mask = latent_mask.cuda()

    sample, intermediates, latent_mask = model.sample_log(
        cond, cond_mask, latent_mask, batch_size=len(cond), sampler="dpm", steps=25,
        log_every_t=200, verbose=False,
    )
    text = model.decode_first_stage(sample, latent_mask, sample_strategy=SAMPLE_STRATEGY['beam'])
    print("\n".join(text))


def gen_intermediate(
        model: LatentDiffusion,
        data_loader,
        steps: int = 25,
        sampler='dpm'
):
    results = []
    intermediates = defaultdict(list)
    with torch.no_grad(), model.ema_scope():
        for batch in track(data_loader, description="Generating..."):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            latent, latent_mask, cond, cond_mask = model.get_input(batch)
            samples, intermediate, latent_mask = model.sample_log(
                cond, cond_mask, latent_mask, batch_size=cond.shape[0], sampler=sampler, steps=steps,
                verbose=False, return_intermediate=True
            )
            finals = model.decode_first_stage(samples, latent_mask, sample_strategy=SAMPLE_STRATEGY['beam'])
            for idx, stage in enumerate(intermediate):
                cur_texts = model.decode_first_stage(stage, latent_mask, sample_strategy=SAMPLE_STRATEGY['beam'])
                intermediates[idx] += cur_texts
            results += finals
    return results, intermediates


def main(opt: argparse.Namespace):
    pl.seed_everything(opt.seed)
    cfg: DictConfig = OmegaConf.load(f"{opt.config}")
    cfg.data.params.batch_size = 512
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = build_dataset(cfg.data)
    model = load_model(cfg.model, opt.ckpt).to(device)
    results, intermediates = gen_intermediate(model, dataset.test_dataloader(), steps=25, sampler='dpm')
    for idx, inter in intermediates.items():
        with open(os.path.join(opt.save_path, f"intermediate_{idx}.txt"), 'w+') as f:
            f.write("\n".join(inter))
    with open(os.path.join(opt.save_path, "final.txt"), 'w+') as f:
        f.write("\n".join(results))


def multi_gen(opt: argparse.Namespace):
    pl.seed_everything(opt.seed)
    cfg: DictConfig = OmegaConf.load(f"{opt.config}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(cfg.model, opt.ckpt).to(device)
    condition = ['what should i do to improve my tennis ?'] * 30
    source = ['what can i do to generally get better at tennis ?'] * 30
    generate(source, condition, model)


if __name__ == '__main__':
    option = parse_args()
    multi_gen(option)
    # main(option)

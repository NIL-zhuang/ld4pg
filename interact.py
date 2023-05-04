import argparse
from typing import List

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import BartTokenizer, BartForConditionalGeneration

from ld4pg.models.diffusion.ddpm import LatentDiffusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/config.yaml", help="path to config which construct model")
    parser.add_argument("--ckpt", type=str, default=None, required=True, help="path to model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="the seed for sampling")
    args = parser.parse_args()
    return args


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
    latent, mask = model.get_first_stage_encoding(src_token['input_ids'], src_token['attention_mask'])
    cond, cond_mask = model.get_conditioning(cond_token['input_ids'], cond_token['attention_mask'])
    return latent, mask, cond, cond_mask


def eval_intermediate(model: LatentDiffusion):
    sentences = [
        "what is the best way to lose weight without exercising?",
        "what can i do to lose weight without doing exercise?"
    ]
    latent, latent_mask, cond, cond_mask = build_input(model, [sentences[0]], [sentences[1]])
    sample, intermediates, latent_mask = model.sample_log(
        cond, cond_mask, latent_mask, batch_size=1,
        sampler="dpm", steps=50, log_every_t=200, verbose=True
    )
    texts = []
    for intermediate in intermediates:
        text = model.decode_first_stage(intermediate, latent_mask)
        texts += text
    print('\n'.join(texts))


def interpolation(model: LatentDiffusion):
    starts = [
        "diet",
        "running",
        "how can i",
        "play basketball",
        "how to lose weight without doing exercise?"
    ]
    srcs = ["what is the best way to lose weight without exercising?"] * len(starts)
    conditions = ["what can i do to lose weight without doing exercise?"] * len(starts)
    latent, latent_mask, cond, cond_mask = build_input(model, srcs, conditions)
    inter_latent, inter_latent_mask, _, _ = build_input(model, starts, conditions)
    # x_start = (latent + inter_latent) / 2
    x_start = inter_latent

    if torch.cuda.is_available():
        model = model.cuda()
        cond = cond.cuda()
        cond_mask = cond_mask.cuda()
        x_start = x_start.cuda()

    sample, intermediates, latent_mask = model.sample_log(
        cond, cond_mask, cond_mask, batch_size=16, sampler="dpm", steps=50,
        log_every_t=200, verbose=True, x_T=x_start
    )
    text = model.decode_first_stage(sample, latent_mask)
    print("\n".join(text))


def main(opt: argparse.Namespace):
    pl.seed_everything(opt.seed)
    cfg: DictConfig = OmegaConf.load(f"{opt.config}")
    model = load_model(cfg.model, opt.ckpt)
    interpolation(model)


if __name__ == '__main__':
    option = parse_args()
    main(option)

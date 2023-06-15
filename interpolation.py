import argparse
from typing import List

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import BartTokenizer, BartForConditionalGeneration

from ld4pg.models.diffusion.ddpm import LatentDiffusion

interp_method = "linear_mean"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/config_qqp.yaml",
                        help="path to config which construct model")
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
    assert noisy_time < 1.0, f"noisy proportion should < 1.0, current is {noisy_time}"
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
        log_every_t=200, verbose=False, x_T=x_start, t_start=noisy_time
    )
    text = model.decode_first_stage(sample, latent_mask)
    print("\n".join(text))


def diffusionEdit(model: LatentDiffusion, alpha=0.4, noisy_time=0.2):
    interps = [
        "can you cook chicken while dancing?",
        "are unicorns real?",
        "can i eat an apple before going to bed?",
        "can playing basketball strengthen your body?",
        "what can i do to lose weight without doing exercise?",
        "how to lose weight without doing exercise?"
    ]
    srcs = ["how can i be strong without too much exercise"] * len(interps)
    conditions = ["what can i do to strengthen myself without doing exercise?"] * len(interps)
    interpolation(srcs, conditions, interps, model, alpha=alpha, noisy_time=noisy_time)


def main(opt: argparse.Namespace):
    pl.seed_everything(opt.seed)
    cfg: DictConfig = OmegaConf.load(f"{opt.config}")
    model = load_model(cfg.model, opt.ckpt)
    for alpha in range(5, 10):
        alpha = alpha / 10
        for noisy_time in range(1, 10):
            noisy_time = noisy_time / 10
            print(f"\n{'=' * 10} alpha: {alpha}, noisy_time: {noisy_time} {'=' * 10}")
            diffusionEdit(model, alpha=alpha, noisy_time=noisy_time)


if __name__ == '__main__':
    option = parse_args()
    main(option)

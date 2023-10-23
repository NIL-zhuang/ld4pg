import argparse
import os
from glob import glob

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizer

from ld4pg.config import SAMPLE_STRATEGY
from ld4pg.data import get_dataset
from ld4pg.data.controlnet_data_module import ControlNetKeywordDataModule
from ld4pg.models.control_net.controlnet import ControlNetModel
from ld4pg.models.control_net.controlnet_inference_pipeline import LDPControlnetInferencePipeline
from ld4pg.models.control_net.controlnet_pipeline import LDPControlNetPipeline
from ld4pg.models.diffusion.ddpm import LatentDiffusion
from ld4pg.util import arg_transform


def build_dataset(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.tokenizer)
    dataset = get_dataset(cfg.name)
    dataset_module = ControlNetKeywordDataModule(
        cfg=cfg.params,
        tokenizer=tokenizer,
        train_dataset=dataset[0],
        valid_dataset=dataset[1],
        test_dataset=dataset[2],
        inf_train_dataloader=False,
    )
    return dataset_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/config_kw_chatgpt.yaml", help="config to construct model")
    parser.add_argument("--ckpt", type=str, default=None, help="path to model checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="path to model checkpoint save dir")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible results)")
    parser.add_argument("--tgt", type=str, default="/home/zhuangzy/result", help="target file dir")
    parser.add_argument("--fname", type=str, default=None, help="target file name")
    parser.add_argument("-u", "--update", nargs='+', default=[], help='update parameters')
    args = parser.parse_args()
    return args


def load_ldp_model(ckpt: str, config: DictConfig):
    diffusion_cfg = config.diffuison.params
    first_stage_model = BartForConditionalGeneration.from_pretrained(diffusion_cfg.enc_dec_model)
    first_stage_tokenizer = BartTokenizer.from_pretrained(diffusion_cfg.enc_dec_model)
    model = LatentDiffusion.load_from_checkpoint(
        ckpt,
        first_stage_model=first_stage_model,
        cond_stage_model=first_stage_model,
        first_stage_tokenizer=first_stage_tokenizer
    )
    model.eval()
    model.freeze()
    return model


def load_controlnet_pipeline(config: DictConfig, base_ldp_path: str, ckpt_path: str):
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

    ldp = load_ldp_model(base_ldp_path, config=config)
    controlnet = build_controlnet_model(base_ldp_path)
    ldp_controlnet_pipeline = LDPControlNetPipeline.load_from_checkpoint(
        ckpt_path,
        text_cond_stage_model=cond_stage_model,
        first_stage_model=first_stage_model,
        tokenizer=first_stage_tokenizer,
        ldp=ldp,
        controlnet=controlnet,
        controlnet_cond_stage_model=control_cond_stage_model.get_encoder(),
    )
    ldp_controlnet_pipeline.freeze()
    ldp_controlnet_pipeline.eval()
    return ldp_controlnet_pipeline


def load_controlnet_inference_pipeline(config: DictConfig, base_ldp_path: str, ckpt_path: str):
    ldp_controlnet_pipeline = load_controlnet_pipeline(config, base_ldp_path, ckpt_path)
    ldp = load_ldp_model(ckpt_path, config)
    inference_pipeline = LDPControlnetInferencePipeline(config, ldp, [ldp_controlnet_pipeline])
    return inference_pipeline


def predict(model, data_loader, step: int = 25):
    results = []
    with torch.no_grad():
        for batch in track(data_loader, description="Generating..."):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            x, x_mask, c, c_mask, cns, cn_masks = model.get_input(batch)
            texts = model.generate_text(
                c, c_mask, x_mask, cns, cn_masks,
                batch_size=c.shape[0], sample_strategy=SAMPLE_STRATEGY['beam1'],
                verbose=False, sampler='dpm', step=step
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
    else:
        raise ValueError("ckpt or ckpt_dir must be specified")

    for m_path in tqdm(sorted(ckpt_list, reverse=True), desc="Evaluating models..."):
        m_name = os.path.splitext(os.path.split(m_path)[-1])[0].split('-')[0]
        if opt.fname is not None:
            m_name += f"-{opt.fname}"
        print(f"generating {m_name}...")

        model: LDPControlnetInferencePipeline = load_controlnet_inference_pipeline(cfg, opt.ckpt, opt.ckpt).to(device)
        model.eval()
        results = predict(model, dataset.test_dataloader(), step=25)
        with open(os.path.join(opt.tgt, f"{m_name}.txt"), 'w+', encoding='utf-8') as f:
            f.write('\n'.join(results))


if __name__ == '__main__':
    options = parse_args()
    main(options)

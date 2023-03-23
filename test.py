import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from ld4pg.dataset.data_module import get_dataset, DataModule
from ld4pg.model.denoising_diffusion import GaussianDiffusion
from ld4pg.utils.arguments import get_inference_parser
from transformers import AutoConfig
from ld4pg.config import *
from ld4pg.model.diffusion_transformer import DiffusionTransformer

ATTN_HEAD_DIM = DEFAULT_DIM_HEAD


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
    # diffusion = GaussianDiffusion(
    #     model,
    #     args,
    #     max_seq_len=model.max_seq_len,
    #     timesteps=args.timesteps,  # number of steps
    #     sampling_timesteps=args.sampling_timesteps,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    #     loss_type=args.loss_type,  # L1 or L2
    #     beta_schedule=args.beta_schedule,
    #     p2_loss_weight_gamma=args.p2_loss_weight_gamma,
    #     objective=args.objective,
    #     ddim_sampling_eta=args.ddim_sampling_eta,
    # )
    # pl_sd = torch.load(args.checkpoint_path, map_location='cpu')
    # diffusion.load_state_dict(pl_sd['state_dict'], strict=False)
    diffusion = GaussianDiffusion.load_from_checkpoint(args.checkpoint_path, model, max_seq_len=64)
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


def build_trainer(args):
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else 1,
        limit_test_batches=0.1
    )
    return trainer


def main():
    parser = get_inference_parser()
    args = parser.parse_args()

    dataset = build_dataset(args)
    trainer = build_trainer(args)
    model = build_model(args)
    model.eval()

    trainer.test(
        model,
        datamodule=dataset,
        # ckpt_path=checkpoint_path
    )


if __name__ == "__main__":
    main()

import argparse


def get_inference_parser():
    parser = get_parser()
    parser.add_argument("--checkpoint_path", type=str, default="/home/zhuangzy/ld4pg.ckpt", help="Path to checkpoint")

    return parser


def get_parser():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default='saved_models')

    parser.add_argument("--dataset_name", type=str, default='qqp')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_token_len", type=int, default=64)

    # Optimization hyperparameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_train_steps", type=int, default=60000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)

    parser.add_argument("--ema_decay", type=float, default=0.9999)
    # parser.add_argument("--ema_update_every", type=int, default=1)

    # Diffusion Hyper parameters
    parser.add_argument("--objective", type=str, default="pred_noise", choices=["pred_noise", "pred_x0"],
                        help="Which parameterization to use for the diffusion objective.")
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l1", "l2", "smooth_l1"],
                        help="Which loss function to use for diffusion.")
    parser.add_argument("--beta_schedule", type=str, default="cosine", choices=["cosine", "linear"],
                        help="Which noise schedule to use.")
    parser.add_argument("--p2_loss_weight_gamma", type=float, default=0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=250)
    parser.add_argument("--normalize_latent", action="store_true", default=False)

    # Generation Arguments
    parser.add_argument("--save_and_sample_every", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--self_condition", action="store_true", default=False)
    parser.add_argument("--ddim_sampling_eta", type=float, default=1)

    # Model hyper parameters
    parser.add_argument("--enc_dec_model", type=str, default="huggingface/bart-base")
    parser.add_argument("--tx_dim", type=int, default=768)
    parser.add_argument("--tx_depth", type=int, default=6)
    parser.add_argument("--scale_shift", action="store_true", default=False)
    parser.add_argument("--disable_dropout", action="store_true", default=False)
    parser.add_argument("--conditional", action="store_true", default=False)
    parser.add_argument("--unconditional_prob", type=float, default=.1)

    return parser

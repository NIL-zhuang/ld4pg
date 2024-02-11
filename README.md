# Latent Diffusion Paraphraser

This is the codebase for the paper Enforcing Paraphrase Generation via Controllable Latent Diffusion.

## Reproduce

### Training

You personal dataset should be placed in `datasets` directory, and split into `train, valid, test` subsets.
Each dataset should be in csv format with `src, tgt` as headers.

When training, you should use `main.py`

* `--config` meaning the path to your yaml config file, which should be placed in `conf` directory
* `--mode` meaning the `train` or `resume` mode
* `--ckpt` is required only in `resume` mode

### Inference

When inference, you should use `seq2seq.py`

* `--ckpt_dir` meaning the checkpoint directory
* `--config` please use the same config file as training, you can find it in `<SAVE_PATH>/conf.yaml`

### Controlnet Ensemble

Use `controlnet_train.py`

* `--ckpt` refers to the original ldp checkpoint path

### Controlnet Inference
* `--ldp` refers to the original ldp checkpoint path
* `--ckpt_dir` meaning the checkpoint directory

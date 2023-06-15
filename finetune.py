import argparse
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW
from transformers import BartForConditionalGeneration, BartTokenizerFast as BartTokenizer

from ld4pg.data.data_module import get_dataset, DataModule
from ld4pg.util import arg_transform

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/config_qqp.yaml", help="path to config which construct model")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible results)")
    parser.add_argument("-u", "--update", nargs='+', default=[], help='update parameters')

    args = parser.parse_args()
    return args


def build_dataset(cfg: DictConfig):
    tokenizer = BartTokenizer.from_pretrained(cfg.params.tokenizer)
    dataset = get_dataset(cfg.name)
    # concat train, valid & test dataset into finetune
    train_dataset = pd.concat([dataset[0], dataset[1], dataset[2]])
    sentences = train_dataset['src'].tolist() + train_dataset['tgt'].tolist()
    new_train = pd.DataFrame({
        "src": sentences,
        "tgt": sentences
    })
    print(new_train.head())
    print(new_train.shape)
    dataset_module = DataModule(
        cfg=cfg.params,
        tokenizer=tokenizer,
        train_dataset=new_train,
        valid_dataset=dataset[1],
        test_dataset=dataset[2],
        inf_train_dataloader=False,
    )
    return dataset_module


def get_save_path(output_dir: str, dataset_name: str, model_name: str):
    local_rank = os.environ.get('LOCAL_RANK', 0)
    if local_rank == 0:
        output_dir = os.path.join(
            output_dir,
            f"{dataset_name}/{model_name}"
        )
        os.makedirs(output_dir, exist_ok=True)
        os.environ['RUN_OUTPUT_DIR'] = output_dir
    else:
        output_dir = os.environ['RUN_OUTPUT_DIR']
    return output_dir


class HFTrainer(pl.Trainer):
    def save_checkpoint(self, filepath, weights_only=False, storage_options=None) -> None:
        if self.is_global_zero:
            dpath = os.path.split(filepath)[0]
            hf_model = self.model.module.module.model
            hf_model.save_pretrained(dpath)


class PLModel(pl.LightningModule):
    def __init__(self, model: BartForConditionalGeneration):
        super().__init__()
        self.model = model

    def compute_loss(self, batch, batch_idx):
        input_ids = batch['source_text_input_ids']
        attention_mask = batch['source_text_attention_mask']
        labels_attention_mask = batch['labels_attention_mask']
        labels = batch['labels']
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=labels_attention_mask
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        loss, logits = self.compute_loss(batch, batch_idx)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self.compute_loss(batch, batch_idx)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-6)
        return optimizer


def build_trainer(cfg, save_path="saved_models"):
    callbacks = [
        RichProgressBar(refresh_rate=1),
        ModelCheckpoint(
            dirpath=save_path,
            filename="{epoch}",
            monitor="val/loss",
            every_n_epochs=1,
            save_top_k=1,
            save_on_train_epoch_end=True
        )
    ]
    trainer = HFTrainer(
        logger=False,
        callbacks=callbacks,
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=-1 if torch.cuda.is_available() else 1,
        precision=32,
        auto_select_gpus=True,
        log_every_n_steps=20,
        strategy=DDPStrategy(find_unused_parameters=False),
        fast_dev_run=False,
        # limit_train_batches=0.01,
    )
    return trainer


def main(opt: argparse.Namespace):
    pl.seed_everything(opt.seed)
    cfg: DictConfig = OmegaConf.load(f"{opt.config}")
    for param in opt.update:
        k, v = param.split("=")
        OmegaConf.update(cfg, k, arg_transform(v), merge=True)

    dataset = build_dataset(cfg.data)
    model_path = cfg.model.diffusion.params.enc_dec_model
    model = BartForConditionalGeneration.from_pretrained(model_path)
    pl_model = PLModel(model)

    save_path = get_save_path("huggingface/finetune", cfg.data.name, os.path.split(model_path)[1])
    print(f"save path: {save_path}")
    trainer = build_trainer(cfg, save_path=save_path)
    trainer.fit(pl_model, dataset)


if __name__ == '__main__':
    option = parse_args()
    main(option)

import csv
import os
from collections import Counter
from pathlib import Path

import pytorch_lightning as pl
import wandb
from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm
from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

import dataset_utils.text_dataset as text_dataset
import diffusion.constant as constant
import diffusion.evaluation as evaluation
import diffusion.optimizer as optimizer
import utils.file_utils as file_utils
from diffusion.utils import *
from utils.torch_utils import compute_grad_norm


class Trainer(object):
    def __init__(
            self,
            args,
            diffusion,
            dataset_name,
            *,
            train_batch_size=16,
            eval_batch_size=64,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            lr_schedule='cosine',
            num_warmup_steps=500,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            adam_weight_decay=0.01,
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            mixed_precision='no',
            split_batches=True,
    ):
        super().__init__()

        pl.seed_everything(42)

        self.args = args

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision,
            log_with='wandb'
        )

        if self.accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})
        self.accelerator.native_amp = amp

        self.diffusion = diffusion

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.max_seq_len = diffusion.max_seq_len

        # Init Encoder-decoder model
        # 一个已经固定好的encoder-decoder模型
        assert 'bart' in args.enc_dec_model
        self.bart_model = BartForConditionalGeneration.from_pretrained(args.enc_dec_model)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.enc_dec_model)

        # dataset and dataloader
        dataset = text_dataset.get_dataset(
            dataset_name,
        )
        self.dataset = dataset.shuffle()
        self.num_samples = min(self.num_samples, len(self.dataset['valid']['text']))
        # Subsample train and val splits for computing language generation during runtime
        self.dataloader = text_dataset.get_dataloader(args, dataset['train'], self.bart_model.config, self.tokenizer, self.max_seq_len)
        self.val_dataloader = text_dataset.get_dataloader(args, dataset['valid'], self.bart_model.config, self.tokenizer, self.max_seq_len)

        # 概率控制长度的采样 样本来自整个数据的长度经验分布
        # 统计整个数据集上每个句子的长度，然后用一个概率来表示采样
        training_lengths = [min(sum(self.dataloader.dataset[idx]['attention_mask']), self.max_seq_len)
                            for idx in range(self.dataloader.dataset.num_rows)]
        length_counts = Counter(training_lengths)
        probs = torch.tensor([length_counts[idx] / self.dataloader.dataset.num_rows for idx in range(self.max_seq_len + 1)])
        assert probs[0] == 0, 'Can\'t have examples of length 0'
        self.length_categorical = torch.distributions.Categorical(probs=probs)

        # 统计类别的种类，用一个概率表示来采样
        if self.diffusion.diffusion_model.class_conditional:
            training_labels = [self.dataloader.dataset[idx]['label'] for idx in range(self.dataloader.dataset.num_rows)]
            label_counts = Counter(training_labels)
            probs = torch.tensor([label_counts[idx] / self.dataloader.dataset.num_rows for idx in range(self.diffusion.diffusion_model.num_classes)])
            self.class_categorical = torch.distributions.Categorical(probs=probs)

        # optimizer and scheduler
        self.opt = optimizer.get_adamw_optimizer(diffusion.parameters(), lr=train_lr, betas=adam_betas, weight_decay=adam_weight_decay)
        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=train_num_steps,
        )

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion, beta=ema_decay, update_every=ema_update_every, power=3 / 4)
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state 状态初始化
        self.step = 0

        # prepare model, dataloader, optim with accelerator
        self.diffusion, self.bart_model, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator. \
            prepare(self.diffusion, self.bart_model, self.opt, self.dataloader, lr_scheduler, self.val_dataloader)
        # 构建 infinite data loop
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)
        self.reference_dict = {}

    def save(self):
        """把模型的运行信息都dump到model.pt中"""
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.diffusion),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, file_path=None):
        """load pretrained model
        加载预训练好的模型
        """
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.diffusion)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        # For backwards compatibility with earlier models
        self.ema.load_state_dict(data['ema'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def log_reference_metrics(self, test=False):
        """ 获取 reference 的评估指标
        """
        train_subset = self.dataset['train']['text'][:self.num_samples]
        train_subset2 = self.dataset['train']['text'][self.num_samples:(2 * self.num_samples)]
        test_subset = self.dataset["test" if test else "valid"]['text'][:self.num_samples]
        for mauve_model_id in ["gpt2-large", "all-mpnet-base-v2"]:
            self.reference_dict[f'reference/{mauve_model_id}_train_train_mauve'], _ = evaluation.compute_mauve(
                train_subset, train_subset2, mauve_model_id)

        train_dataset = self.dataset['train']['text']
        if test:
            # train_subset = self.dataset['train']['text'][:self.num_samples]
            # train_subset2 = self.dataset['train']['text'][self.num_samples:(2 * self.num_samples)]
            # test_subset = self.dataset['test']['text'][:self.num_samples]

            for mauve_model_id in ["gpt2-large", "all-mpnet-base-v2"]:
                self.reference_dict[f'reference/{mauve_model_id}_train_test_mauve'], _ = evaluation.compute_mauve(
                    train_subset, test_subset, mauve_model_id)
            for k, v in evaluation.compute_diversity(test_subset):
                self.reference_dict[f"reference/test_{k}"] = v
            self.reference_dict['reference/test_perplexity'] = evaluation.compute_perplexity(test_subset)
            self.reference_dict[f"reference/test_memorization"] = evaluation.compute_memorization(test_subset, train_dataset)
            self.reference_dict['reference/test_unique_wordcount'] = evaluation.compute_wordcount(test_subset)
            return

        # test_subset = self.dataset['valid']['text'][:self.num_samples]
        # train_subset = self.dataset['train']['text'][:self.num_samples]
        # train_subset2 = self.dataset['train']['text'][self.num_samples:(2 * self.num_samples)]
        self.reference_dict['reference/train_unique_wordcount'] = evaluation.compute_wordcount(train_subset)
        self.reference_dict['reference/train_perplexity'] = evaluation.compute_perplexity(train_subset)
        for k, v in evaluation.compute_diversity(train_subset).items():
            self.reference_dict[f"reference/train_{k}"] = v

        for mauve_model_id in ["gpt2-large", "all-mpnet-base-v2"]:
            self.reference_dict[f'reference/{mauve_model_id}_train_val_mauve'], _ = evaluation.compute_mauve(
                train_subset, test_subset, mauve_model_id)
        for k, v in evaluation.compute_diversity(test_subset).items():
            self.reference_dict[f"reference/val_{k}"] = v
        self.reference_dict['reference/val_perplexity'] = evaluation.compute_perplexity(test_subset)
        self.reference_dict[f"reference/val_memorization"] = evaluation.compute_memorization(test_subset, train_dataset)
        self.reference_dict['reference/val_unique_wordcounts'] = evaluation.compute_wordcount(test_subset)
        # torch.cuda.empty_cache()

    @torch.no_grad()
    def gen_synthetic_dataset(self, num_samples, seed=42):
        """用于test合成数据集"""
        num_classes = self.diffusion.diffusion_model.num_classes
        num_samples_per_class = num_samples // num_classes
        assert num_samples % num_classes == 0, f'Dataset size ({num_samples}) must be divisible by the number of classes ({num_classes})'
        data = {'text': [], 'label': []}
        self.ema.ema_model.eval()
        torch.manual_seed(seed)
        device = self.accelerator.device
        for class_id in range(num_classes):
            text = []
            while len(text) < num_samples_per_class:
                batches = num_to_groups(num_samples_per_class - len(text), self.eval_batch_size)
                model_outputs = list(
                    map(lambda n: tuple(
                        x.to('cpu')
                        for x in self.ema.ema_model.sample(
                            batch_size=n,
                            length=self.length_categorical.sample((n,)),
                            class_id=torch.tensor([class_id] * n, dtype=torch.long, device=device)
                        )
                    ), batches)
                )

                for (latents, mask) in model_outputs:
                    latents, mask = latents.to(device), mask.to(device)
                    if self.args.normalize_latent:
                        latents = self.ema.ema_model.denormalize_latent(latents)
                    encoder_output = BaseModelOutput(last_hidden_state=latents.clone())
                    sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask.clone(),
                                                          **constant.generate_kwargs['beam'])
                    texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
                    texts_list = [text.strip() for text in texts_list if len(text.strip()) > 0]
                    text.extend(texts_list)
            data['text'].extend(text)
            data['label'].extend([class_id] * num_samples_per_class)

        save_path = os.path.join(self.results_folder, f'synth_sample{num_samples}_seed{seed}.csv')
        print(save_path)
        with open(save_path, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))

    @torch.no_grad()
    def sample(self, num_samples=None, class_id=None, seed=42, test=False):
        num_samples = default(num_samples, self.num_samples)
        device = self.accelerator.device
        self.diffusion.to('cpu')
        # torch.cuda.empty_cache()

        self.ema.ema_model.eval()

        num_samples, reference_texts = self.extract_sample_references(class_id, num_samples, test)
        # Stores generation outputs for each strategy
        # beam采样和核采样
        all_texts_lists = {k: [] for k, _ in constant.generate_kwargs.items()}

        torch.manual_seed(seed)

        def get_class_id(n):
            # 生成一系列class_id
            # 如果设置了class_unconditional_prob，那么就是 num_classes均匀
            # 否则就是从类型分布中采样
            if exists(class_id):
                # return torch.full((n,), class_id, dtype=torch.long, device=device)
                return torch.tensor([class_id] * n, dtype=torch.long, device=device)
            if self.diffusion.diffusion_model.class_conditional:
                if self.diffusion.diffusion_model.unconditional_prob > 0:
                    return torch.tensor([self.diffusion.diffusion_model.num_classes] * n, dtype=torch.long, device=device)
                return self.class_categorical.sample((n,)).to(device)
            return None

        # Loop until enough sentences have been generated across all strategies
        # 获取 num_sample 个采样
        while min([len(all_texts_lists[ele]) for ele in all_texts_lists]) < num_samples:
            # 获取一系列 batch
            batches = num_to_groups(
                num_samples - min([len(all_texts_lists[ele]) for ele in all_texts_lists]),
                max(self.eval_batch_size, self.train_batch_size)
            )
            # 把 batch 传给 ema_model 进行sample，得到 latents 和 mask
            model_outputs = list(map(
                lambda n: tuple(
                    x.to('cpu')
                    for x in self.ema.ema_model.sample(
                        batch_size=n,
                        length=self.length_categorical.sample((n,)),
                        class_id=get_class_id(n)
                    )),
                batches
            ))

            for (latents, mask) in model_outputs:
                latents, mask = latents.to(device), mask.to(device)
                if self.args.normalize_latent:
                    latents = self.ema.ema_model.denormalize_latent(latents)
                for k, kwargs in constant.generate_kwargs.items():
                    encoder_output = BaseModelOutput(last_hidden_state=latents.clone())
                    sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask.clone(), **kwargs)
                    texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
                    texts_list = [text.strip() for text in texts_list if len(text.strip()) > 0]
                    all_texts_lists[k].extend(texts_list)

        assert min([len(all_texts_lists[ele]) for ele in all_texts_lists]) >= num_samples
        text_generations = {k: v[:num_samples] for k, v in all_texts_lists.items()}

        # 评估生成文本质量
        self.ema.to('cpu')
        # torch.cuda.empty_cache()
        class_id_prefix = f'cond{class_id}_' if exists(class_id) else ''
        metrics = self.evaluate_sample_quality(class_id_prefix, reference_texts, seed, text_generations)

        if len(self.reference_dict) == 0 or test:
            self.log_reference_metrics(test)
        if test:
            metrics_dict = {**metrics, **self.reference_dict}
            metrics_dict = {f'{k}_seed{seed}': v for k, v in metrics_dict.items()}
            self.accelerator.log(metrics_dict, self.step)
        else:
            self.accelerator.log({**metrics, **self.reference_dict}, self.step)
        # torch.cuda.empty_cache()
        self.diffusion.to(device)
        self.ema.to(device)

    def extract_sample_references(self, class_id, num_samples, test):
        """构建sample的reference数据集，取其中的num_samples个数据"""
        reference_texts = {}
        if exists(class_id):
            # conditional generation
            for filter_class_id in range(self.diffusion.diffusion_model.num_classes):
                filtered_dataset = self.dataset.filter(lambda example: example["label"] == filter_class_id)
                if test:
                    reference_texts[f'ref{filter_class_id}_test'] = filtered_dataset['test']['text']
                else:
                    reference_texts[f'ref{filter_class_id}_val'] = filtered_dataset['valid']['text']
                    reference_texts[f'ref{filter_class_id}_train'] = filtered_dataset['train']['text']

            for key, reference_text in reference_texts.items():
                num_samples = min(num_samples, len(reference_text))
            reference_texts = {k: v[:num_samples] for k, v in reference_texts.items()}
        else:
            if test:
                reference_texts[f'test'] = self.dataset['test']['text'][:num_samples]
                reference_texts['train'] = self.dataset['train']['text'][:num_samples]
            else:
                reference_texts['val'] = self.dataset['valid']['text'][:num_samples]
                reference_texts['train'] = self.dataset['train']['text'][:num_samples]
        return num_samples, reference_texts

    def evaluate_sample_quality(self, class_id_prefix, reference_texts, seed, text_generations):
        """生成文本质量评估
        评估 beam 和 核采样 两种采样strategy 产出的文本质量
        """
        metrics = {}
        milestone = self.step // self.save_and_sample_every
        for strategy, all_texts_list in text_generations.items():
            table = wandb.Table(columns=['Samples'], data=[[text] for text in all_texts_list])
            self.accelerator.log({f"{strategy}/{class_id_prefix}samples": table}, self.step)

            file_utils.save_text_samples(
                all_texts_list, os.path.join(
                    self.results_folder,
                    f'{"eval-" if self.args.eval else ""}{f"eval{seed}-" if self.args.eval_test else ""}'
                    f'{class_id_prefix}{strategy}-sample-{milestone}.txt'
                ))
            metrics[f"{strategy}/{class_id_prefix}perplexity"] = evaluation.compute_perplexity(all_texts_list)
            metrics[f"{strategy}/{class_id_prefix}unique_wordcount"] = evaluation.compute_wordcount(all_texts_list)
            for k, v in evaluation.compute_diversity(all_texts_list).items():
                metrics[f"{strategy}/{class_id_prefix}{k}"] = v
            metrics[f"{strategy}/{class_id_prefix}memorization"] = evaluation.compute_memorization(all_texts_list, self.dataset['train']['text'])

            # Only evaluate MAUVE if generations are reasonable
            # 如果产出文本的PPL太高，说明在说胡话，就不测MAUVE了
            if metrics[f"{strategy}/{class_id_prefix}perplexity"] > 5000:
                continue

            for mauve_model_id in ["gpt2-large", "all-mpnet-base-v2"]:
                for key, reference_text in reference_texts.items():
                    # 分别和 reference 的 train, val, test 集合比较 MAUVE 结果
                    metrics[f"{strategy}/{mauve_model_id}_{class_id_prefix}{key}_mauve"], _ = evaluation.compute_mauve(
                        all_texts_list, reference_text, mauve_model_id)
        return metrics

    @staticmethod
    def evaluate_metric(hypos, srcs, refs, mauve=True):
        metrics = {}
        MAUVE_MODELS = ["gpt2-large", "all-mpnet-base-v2"]
        metrics['perplexity'] = evaluation.compute_perplexity(hypos)
        for k, v in evaluation.compute_diversity(hypos).items():
            metrics[f"diversity_{k}"] = v
        metrics["memorization"] = evaluation.compute_memorization(hypos, srcs)
        metrics["unique_wordcount"] = evaluation.compute_wordcount(hypos)

        # Only evaluate MAUVE if generations are reasonable
        # 如果产出文本的PPL太高，说明在说胡话，就不测MAUVE了
        if metrics['perplexity'] > 5000 and not mauve:
            return metrics

        for mauve_model_id in MAUVE_MODELS:
            for key, ref in refs.items():
                metrics[f"{mauve_model_id}_{key}_mauve"], div_curve = evaluation.compute_mauve(hypos, ref, mauve_model_id)
        return metrics

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                # TODO center and normalize BART latent space with empirical est. of mean/var.

                # train loop
                data = next(self.data_iter).to(device)
                total_loss = self.train_loop(data)

                accelerator.wait_for_everyone()
                grad_norm = compute_grad_norm(self.diffusion.parameters())
                accelerator.clip_grad_norm_(self.diffusion.parameters(), 1.0)
                if not self.args.resume_training:
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                accelerator.wait_for_everyone()
                self.step += 1

                # validate process
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    # Logging
                    if self.step % 50 == 0:
                        self.diffusion.eval()
                        self.ema.ema_model.eval()
                        with torch.no_grad():
                            data = next(self.val_iter).to(device)
                            total_val_ema_loss, total_val_loss = self.calculate_valid_loss(data)

                        logs = {
                            "loss": total_loss,
                            "val_loss": total_val_loss,
                            "val_ema_loss": total_val_ema_loss,
                            "grad_norm": grad_norm,
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "step": self.step,
                            "epoch": (self.step * self.gradient_accumulate_every) / len(self.dataloader),
                            "samples": self.step * self.train_batch_size * self.gradient_accumulate_every
                        }
                        pbar.set_postfix(**logs)
                        accelerator.log(logs, step=self.step)

                        self.diffusion.train()

                    # sample and valid
                    if self.step % self.save_and_sample_every == 0:
                        self.diffusion.eval()
                        self.sample()
                        if self.diffusion.diffusion_model.class_conditional:
                            for class_id in range(self.diffusion.diffusion_model.num_classes):
                                if self.args.dataset_name == 'ag_news':
                                    num_samples = 100
                                elif self.args.dataset_name == 'sst':
                                    num_samples = 500
                                self.sample(num_samples=num_samples, class_id=class_id)
                        self.save()
                        self.diffusion.train()

                pbar.update(1)

        accelerator.print('training complete')

    def calculate_valid_loss(self, data):
        total_val_loss = 0.
        total_val_ema_loss = 0.
        for grad_accum_step in range(self.gradient_accumulate_every):
            latent = self.bart_model.get_encoder()(input_ids=data['input_ids'], attention_mask=data['attention_mask']).last_hidden_state
            if self.args.normalize_latent or self.args.scale_latent:
                latent = self.diffusion.normalize_latent(latent)

            mask = data['attention_mask'].bool()
            with self.accelerator.autocast():
                loss = self.diffusion(latent, mask, class_id=(data['label'] if self.diffusion.diffusion_model.class_conditional else None))
                loss = loss / self.gradient_accumulate_every
                total_val_loss += loss.item()
                loss = self.ema.ema_model(latent, mask, class_id=(data['label'] if self.diffusion.diffusion_model.class_conditional else None))
                loss = loss / self.gradient_accumulate_every
                total_val_ema_loss += loss.item()
        return total_val_ema_loss, total_val_loss

    def train_loop(self, data):
        total_loss = 0.
        for grad_accum_step in range(self.gradient_accumulate_every):
            with torch.no_grad():
                latent = self.bart_model.get_encoder()(input_ids=data['input_ids'], attention_mask=data['attention_mask']).last_hidden_state
                if self.args.normalize_latent:
                    # 这个归一化的方差是每个维度上的方差还是单纯的标准差？
                    if self.step == 0 and grad_accum_step == 0:
                        # 最一开始设置隐变量归一化的均值和方差
                        self.init_normalize_latent(data, latent)
                    latent = self.diffusion.normalize_latent(latent)

            mask = data['attention_mask'].bool()
            with self.accelerator.autocast():
                loss = self.diffusion(latent, mask, class_id=(data['label'] if self.diffusion.diffusion_model.class_conditional else None))
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()

            self.accelerator.backward(loss)
        return total_loss

    def init_normalize_latent(self, data, latent):
        # todo: 方差计算存在问题，用BatchNorm进行替代
        # 这里的latent_mean是768的，但sigma是全局的，而且不会进行参数化更新
        # latent_mean: 一个batch里每个维度的均值
        # latent_scale: 标准差
        latent_vector = torch.cat([latent[i][:torch.sum(data['attention_mask'][i])] for i in range(latent.shape[0])], dim=0)
        # Add mean stats to model and EMA wrapper
        self.diffusion.latent_mean = torch.mean(latent_vector, dim=0)
        self.ema.ema_model.latent_mean = self.diffusion.latent_mean
        # Add var stats to model and EMA wrapper
        self.diffusion.latent_scale = torch.std(latent_vector - self.diffusion.latent_mean, unbiased=False)
        self.ema.ema_model.latent_scale = self.diffusion.latent_scale

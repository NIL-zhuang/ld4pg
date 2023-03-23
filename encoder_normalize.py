import pandas as pd
import torch
from rich.progress import track
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, AutoTokenizer

from ld4pg.dataset.data_module import get_dataset


class DatasetModule(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.source = tokenizer(
            data,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            add_special_tokens=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return dict(
            input_ids=self.source['input_ids'][index],
            attention_mask=self.source['attention_mask'][index]
        )


def main():
    model_path = "huggingface/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = pd.concat(get_dataset("qqp"))
    sentences = dataset['src'].tolist() + dataset['tgt'].tolist()
    print(len(sentences))

    input_dataset = DatasetModule(data=sentences[:4096], tokenizer=tokenizer)
    dataloader = DataLoader(input_dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BartForConditionalGeneration.from_pretrained(model_path)
    encoder = model.get_encoder().to(device)

    all_latents = []

    for idx, batch in enumerate(track(dataloader)):
        src = batch['input_ids'].to(device)
        src_mask = batch['attention_mask'].to(device)
        if idx % 100 == 0:
            print(f"{idx}: Fuck this shit!")
        with torch.no_grad():
            latent = encoder(src, attention_mask=src_mask).last_hidden_state
        masked_latent = torch.masked_select(latent, ~src_mask.unsqueeze(-1).expand(latent.shape).bool())
        all_latents.append(masked_latent.cpu())

    all_latents_tensor = torch.cat(all_latents)
    print(all_latents_tensor.shape)
    std = all_latents_tensor.std().item()
    mean = all_latents_tensor.mean().item()
    print(f"std: {std}")
    print(f"mean: {mean}")
    print(f"inverse std: {1 / std}")


if __name__ == '__main__':
    main()

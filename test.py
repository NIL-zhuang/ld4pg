from transformers import BartTokenizer, BartForConditionalGeneration, BartPretrainedModel
import torch
from transformers.modeling_outputs import BaseModelOutput

MODEL_PATH = "huggingface/bart-base"


def interpolation(s1: str, s2: str, model, tokenizer, steps: int = 20):
    s = tokenizer([s1, s2], padding=True, return_tensors="pt")
    mask = s['attention_mask'][1:]
    latent = model.get_encoder()(**s).last_hidden_state
    l1, l2 = latent[0], latent[1]
    diff = (l2 - l1) / steps

    result = []
    for step in range(steps):
        inter = (l1 + diff * step).unsqueeze(0)
        output = BaseModelOutput(last_hidden_state=inter.clone())
        samples = model.generate(
            encoder_outputs=output,
            attention_mask=mask.clone(),
            max_length=64,
            min_length=5,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
        text_list = [
            tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in samples
        ]
        text_list = [
            text.strip() for text in text_list
            if len(text.strip()) > 0
        ]
        result += text_list
    for idx, res in enumerate(result):
        print(idx, res)


def emb_init(model: BartPretrainedModel):
    encoder = model.get_encoder()
    print(encoder.embed_tokens)
    print(encoder.embed_tokens.weight.shape)
    init_val = torch.mean(encoder.embed_tokens.weight, dim=0)
    print(init_val.shape)
    print(init_val)


def main():
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    emb_init(model)
    # s1 = "My name is Huang, and I love NJU."
    # s2 = "I love NJU, Huang is my name."
    # interpolation(s1, s2, model, tokenizer)


if __name__ == '__main__':
    main()

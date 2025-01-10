import argparse
import os
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import peft

MAX_LENGTH = 2048

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_embeddings_in_batches(model, tokenizer, texts, max_length, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i : i + batch_size]
        batch_dict = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(**batch_dict)
            batch_embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1).cpu()
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)


def load_model_and_tokenizer(base_model_path, lora_path, load_in_4bit=True):
    model = AutoModel.from_pretrained(
        base_model_path,
        device_map=0,
        torch_dtype=torch.float16,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path if lora_path else base_model_path
    )
    model.resize_token_embeddings(len(tokenizer))
    if lora_path:
        model = peft.PeftModel.from_pretrained(model, lora_path)
    return model, tokenizer


def main(args):
    output_file = args.input_text.replace(
        ".json", ".pt.fold.{}.{}.embed".format(*args.fold)
    )
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping...")
        return
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.lora_path, load_in_4bit=args.load_in_4bit
    )
    texts = json.load(open(args.input_text))["texts"][args.fold[0] :: args.fold[1]]
    embeddings = get_embeddings_in_batches(
        model,
        tokenizer,
        texts,
        max_length=MAX_LENGTH,
        batch_size=4,
    )
    text2embeds = {text: emb for text, emb in zip(texts, embeddings)}
    torch.save(text2embeds, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Path to the base model",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the LoRA model",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default=".cache/data.json",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit mode",
    )
    parser.add_argument("--fold", nargs=2, type=int, default=[0, 1])
    args = parser.parse_args()
    if not os.path.exists(args.lora_path):
        args.lora_path = None
    main(args)

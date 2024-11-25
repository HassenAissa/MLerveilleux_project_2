import numpy as np
import datasets
from gpt import GPTBase
from moe import TimeDependantMoE2
import torch
import os
from multiprocessing import cpu_count
import tiktoken
from datasets import load_dataset
from datetime import datetime

from config import Config

SEED = 42
FINEWEB_DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets/fineweb/")
num_proc = max(4, cpu_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = tiktoken.get_encoding("gpt2")

def process_data(example):
    text_tokens = tokenizer.encode_ordinary(example["text"])
    text_tokens.append(tokenizer.eot_token)
    return {"tokens": text_tokens, "date": example["date"]}

def get_fineweb_dataset(num_proc=num_proc):
    dataset = load_dataset(path="HuggingFaceFW/fineweb", name="sample-10BT", cache_dir="huggingface_cache/datasets")
    split_dataset = dataset["train"].train_test_split(test_size=0.005, seed=SEED, shuffle=True)
    split_dataset["test"] = split_dataset.pop("test")

    tokenized = split_dataset.map(
        process_data,
        remove_columns=["text"],
        desc="Tokenizing the splits",
        num_proc=num_proc,
    )
    return tokenized



fineweb_dataset = get_fineweb_dataset()

print("Dataset loaded")

max_date = int(max(max(fineweb_dataset["train"]["date"]), max(fineweb_dataset["test"]["date"]))[:4])
min_date = int(min(min(fineweb_dataset["train"]["date"]), min(fineweb_dataset["test"]["date"]))[:4])
date_list = [datetime(year, 1, 1) for year in range(min_date, max_date + 1)]
config = Config(**{
    "moe_num_experts": len(date_list),
    "moe_softmax_order": "softmax_topk",
    "batch_size": 4,
    "n_embd": 768,
    "date_list": date_list,
})

print("Creating model")

moe = GPTBase(config)
print("Model created")
moe.to(device)


optimizer = torch.optim.Adam(moe.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

print("Starting training")
for epoch in range(1):
    for i in range(0, len(fineweb_dataset["train"]), config.batch_size):
        batch = fineweb_dataset["train"][i:i+config.batch_size]
        # make all batch["tokens"] the same length
        max_len = max([len(tokens) for tokens in batch["tokens"]])
        for tokens in batch["tokens"]:
            print("toktok")
            tokens += [0] * (max_len - len(tokens))
        batch["tokens"] = torch.tensor(batch["tokens"]).to(device)
        batch["date"] = np.array(batch["date"])
        
        optimizer.zero_grad()
        output = moe(batch["tokens"], batch["date"])
        loss = criterion(output, batch["tokens"])
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


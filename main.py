import numpy as np
import datasets
from gpt import GPTBase
from moe import TimeDependantMoE2
import torch
import os
from multiprocessing import cpu_count
import tiktoken
from datasets import load_dataset
import numpy as np

# Load the dataset
SEED = 42
FINEWEB_DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets/fineweb/")
num_proc = max(4, cpu_count())
tokenizer = tiktoken.get_encoding("gpt2")


def process_data(example):
    text_tokens = tokenizer.encode_ordinary(example["text"])
    text_tokens.append(tokenizer.eot_token)
    return {"tokens": text_tokens, "date": example["date"]}

def get_fineweb_dataset(num_proc=num_proc):

    # Check if train.bin and test.bin already exist
    train_bin_path = os.path.join(FINEWEB_DATASET_PATH, "train.bin")
    test_bin_path = os.path.join(FINEWEB_DATASET_PATH, "test.bin")
    
    if not os.path.exists(train_bin_path) or not os.path.exists(test_bin_path):
        os.makedirs(FINEWEB_DATASET_PATH, exist_ok=True)

        dataset = load_dataset(path="HuggingFaceFW/fineweb", name="sample-10BT", cache_dir="huggingface_cache/datasets")
        split_dataset = dataset.train_test_split(test_size=0.005, seed=SEED, shuffle=True)
        split_dataset["test"] = split_dataset.pop("test")

        tokenized = split_dataset.map(
            process_data,
            remove_columns=["text"],
            desc="Tokenizing the splits",
            num_proc=num_proc,
        )
    return tokenized



fineweb_dataset = get_fineweb_dataset()

config = {
    "moe_num_experts": 4,
    "moe_softmax_order": "softmax_topk",
    "batch_size": 4,
    "n_embd": 256,
}
mlp = GPTBase
moe = TimeDependantMoE2(config, mlp)

optimizer = torch.optim.Adam(moe.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()   

for epoch in range(1):
    for i in range(0, len(fineweb_dataset["train"]), config["batch_size"]):
        batch = fineweb_dataset["train"][i:i+config["batch_size"]]
        # make all batch["tokens"] the same length
        max_len = max([len(tokens) for tokens in batch["tokens"]])
        for tokens in batch["tokens"]:
            tokens += [0] * (max_len - len(tokens))
        batch["tokens"] = torch.tensor(batch["tokens"])
        batch["date"] = torch.tensor(batch["date"])
        
        optimizer.zero_grad()
        output = moe(batch["tokens"], batch["date"])
        loss = criterion(output, batch["tokens"])
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


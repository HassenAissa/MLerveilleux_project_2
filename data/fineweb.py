from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import os
from multiprocessing import cpu_count

SEED = 42
FINEWEB_DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets/fineweb/")

# Adjust the number of processes based on available CPU cores
num_proc = min(4, cpu_count())

tokenizer = tiktoken.get_encoding("gpt2")


def process_data(example):
    tokens = tokenizer.encode_ordinary(example["text"])
    tokens.append(tokenizer.eot_token)
    return {"tokens": tokens, "len": len(tokens)}


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

        # Save the tokenized datasets to memory-mapped files
        train_array = np.array(tokenized["train"]["tokens"], dtype=np.uint16)
        test_array = np.array(tokenized["test"]["tokens"], dtype=np.uint16)

        train_memmap = np.memmap(train_bin_path, dtype=np.uint16, mode="w+", shape=train_array.shape)
        test_memmap = np.memmap(test_bin_path, dtype=np.uint16, mode="w+", shape=test_array.shape)

        train_memmap[:] = train_array[:]
        test_memmap[:] = test_array[:]

        train_memmap.flush()
        test_memmap.flush()

    # Load the memory-mapped files
    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
    test_data = np.memmap(test_bin_path, dtype=np.uint16, mode="r")

    return {"train": train_data, "test": test_data}


if __name__ == "__main__":
    fineweb_dataset = get_fineweb_dataset()
    print(f"Dataset loaded with {len(fineweb_dataset['train'])} training tokens and {len(fineweb_dataset['test'])} test tokens.")

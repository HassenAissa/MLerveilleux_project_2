from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import os

SEED = 42
FINEWEB_DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets/fineweb/")

tokenizer = tiktoken.get_encoding("gpt2")


def process_data(example):
    tokens = tokenizer.encode_ordinary(example["text"])
    tokens.append(tokenizer.eot_token)
    return {"tokens": tokens, "len": len(tokens)}


def get_fineweb_dataset(num_proc=8):

    # build the dataset if it doesn't exist
    if not os.path.exists(os.path.join(FINEWEB_DATASET_PATH, "train.bin")):
        os.makedirs(FINEWEB_DATASET_PATH, exist_ok=True)

        dataset = load_dataset(path="HuggingFaceFW/fineweb", name="sample-10BT", cache_dir="huggingface_cache/datasets")

        split_dataset = dataset["train"].train_test_split(test_size=0.005, seed=SEED, shuffle=True)
        split_dataset["test"] = split_dataset.pop("test")
        print(split_dataset)

        tokenized = split_dataset.map(
            process_data,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the tokens in each dataset into one large file we can use for training
        for split, data_split in tokenized.items():
            arr_len = np.sum(data_split["len"])
            filename = os.path.join(FINEWEB_DATASET_PATH, f"{split}.bin")
            arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,)) # (max_token_value = 50256 < 2**16 ==> uint16)
            total_batches = min(1024, len(data_split))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                batch = data_split.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy") # batch for quicker writes
                arr_batch = np.concatenate(batch["tokens"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    train_data = np.memmap(
        os.path.join(FINEWEB_DATASET_PATH, "train.bin"), dtype=np.uint16, mode="r"
    )
    test_data = np.memmap(
        os.path.join(FINEWEB_DATASET_PATH, "test.bin"), dtype=np.uint16, mode="r"
    )

    return {"train": train_data, "test": test_data}


if __name__ == "__main__":
    get_fineweb_dataset()
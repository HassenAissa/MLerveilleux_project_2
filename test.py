import datasets
from gpt import GPTBase
import torch
import os
from multiprocessing import cpu_count
import tiktoken
from datasets import load_dataset

from config import Config
from tqdm import tqdm
import json
from utils import save_checkpoint, print_model_architecture

torch.set_default_dtype(torch.bfloat16)

SEED = 42
FINEWEB_DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets/fineweb/")
num_proc = max(4, cpu_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = tiktoken.get_encoding("gpt2")
# CHANGE THIS WITH PATH OF MODEL
CHECKPOINTS_PATH = ["model1.pth", "model2.pth"]
# CHANGE THIS !!! !
SEQUENCE_LENGTH = 1024

def process_data(example, min_date, max_date):
    text_tokens = tokenizer.encode_ordinary(example["text"])
    text_tokens.append(tokenizer.eot_token)
    date = int(example["date"][:4])
    mask_date = torch.zeros(max_date - min_date + 1)
    mask_date[:date - min_date + 1] = 1
    return {"tokens": text_tokens, "date": mask_date}


def get_fineweb_dataset(num_proc=num_proc):
    dataset = load_dataset(path="HuggingFaceFW/fineweb", name="sample-10BT", cache_dir="huggingface_cache/datasets")
    split_dataset = dataset["train"].train_test_split(test_size=0.005, seed=SEED, shuffle=True)

    min_date = int(min(min(split_dataset["train"]["date"]), min(split_dataset["test"]["date"]))[:4])
    max_date = int(max(max(split_dataset["train"]["date"]), max(split_dataset["test"]["date"]))[:4])

    tokenized = split_dataset.map(
        process_data,
        remove_columns=["text"],
        desc="Tokenizing the splits",
        num_proc=num_proc,
        fn_kwargs={"min_date": min_date, "max_date": max_date},
    )
    return tokenized, min_date, max_date


fineweb_dataset, min_date, max_date = get_fineweb_dataset()
print("Dataset loaded")
moe_routings = [None, "standard_gating", "masked"]


for path, moe_routing, id in zip(CHECKPOINTS_PATH, moe_routings, range(len(moe_routings))):
    print("Testing model #", id)
    config = Config(**{
        "moe_num_experts": max_date - min_date + 1,
        "moe_softmax_order": "softmax_topk",
        "batch_size": 64,
        "n_embd": 768,
        # "date_list": date_list,
        "moe_routing": moe_routing,
        "moe": moe_routing is not None
    })

    moe = GPTBase(config)

    moe.load_state_dict(torch.load(path)["model"])
    moe.to(device)


    moe.eval()
    print("Model loaded !")



    with torch.no_grad():
        batch = fineweb_dataset["test"]
        # make all batch["tokens"] the same length
        max_len = config.sequence_length
        batch["tokens"] = [tokens[:max_len] for tokens in batch["tokens"]]
        for tokens in batch["tokens"]:
            tokens += [0] * (max_len - len(tokens))

        batch["tokens"] = torch.tensor(batch["tokens"]).to(device)
        batch["date"] = torch.tensor(batch["date"]).to(device)

        output = moe(batch["tokens"], batch["date"],
                     targets=batch["tokens"], get_logits=False, moe=True)
        # output = {"logits": logits, "loss": loss, "aux_losses": aux_losses, "router_logits": router_logits,}
        # loss = criterion(output, batch["tokens"])
        loss = output["loss"]
        print("Loss on the Test Set : ", loss)
        print("Perplexity on the Test Set : ", torch.exp(loss))
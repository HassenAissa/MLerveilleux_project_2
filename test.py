

CHECKPOINTS_PATH = ["../../haissa/MLerveilleux_project_2/best_model_None.pth","best_model_None.pth", "best_model_standard_gating.pth", "best_model_masked.pth"]
import math 
SEQUENCE_LENGTH = 1024
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

basic_config = Config(**{
    "moe_num_experts": 10, 
    "moe_softmax_order": "softmax_topk",
    "batch_size": 500,
    "n_embd": 768,
    "moe_routing": None,
    "moe": False
})

def process_data(example, min_date, max_date):
    text_tokens = tokenizer.encode_ordinary(example["text"])
    text_tokens.append(tokenizer.eot_token)
    date = int(example["date"][:4])
    mask_date = torch.zeros((max_date - min_date + 1)//2)
    mask_date[:(date - min_date + 1)//2] = 1
    max_len = basic_config.sequence_length
    text_tokens = text_tokens[:max_len]
    text_tokens += [tokenizer.eot_token] * (max_len - len(text_tokens))
    return {"tokens": text_tokens, "date": mask_date}

def get_fineweb_dataset(num_proc=num_proc):
    dataset = load_dataset(path="HuggingFaceFW/fineweb", name="sample-10BT", cache_dir="../../lcostes/MLerveilleux_project_2/huggingface_cache/datasets")
    split_dataset = dataset["train"].train_test_split(test_size=0.005, seed=SEED, shuffle=True)
    
    min_date = int(min(min(split_dataset["train"]["date"]), min(split_dataset["test"]["date"]))[:4])
    max_date = int(max(max(split_dataset["train"]["date"]), max(split_dataset["test"]["date"]))[:4])
    
    tokenized = split_dataset["test"].map(
        process_data,
        remove_columns=["text"],
        desc="Tokenizing the splits",
        num_proc=num_proc,
        fn_kwargs={"min_date": min_date, "max_date": max_date},
    )
    return tokenized, min_date, max_date


print("Starting to load datase")
fineweb_dataset, min_date, max_date = get_fineweb_dataset()
print("Dataset loaded")
moe_routings = [None, None, "standard_gating", "masked"]


for path, moe_routing, id in zip(CHECKPOINTS_PATH, moe_routings, range(len(moe_routings))):
    print("Testing model #", id)
    print("Path : ", path)
    print("Type : ", moe_routing)
    config = Config(**{
        "moe_num_experts": (max_date - min_date + 1)//2,
        "moe_softmax_order": "softmax_topk",
        "batch_size": 64,
        "n_embd": 768,
        # "date_list": date_list,
        "moe_routing": moe_routing,
        "moe": moe_routing is not None
    })

    moe = GPTBase(config)
    state_dict = torch.load(path)["model"]
    moe.load_state_dict(state_dict)
    moe.to(device)


    moe.eval()
    print("Model loaded !")
    nb_points = len(fineweb_dataset)
    print("NB : ", nb_points)


    with torch.no_grad():
        print("taking whole test batch")
        
        # make all batch["tokens"] the same length
        max_len = config.sequence_length
        loss_sum = 0
        for i in tqdm(range(0, nb_points, config.batch_size)):
            batch = fineweb_dataset[i:i+config.batch_size]
            batch["tokens"] = [tokens[:max_len] for tokens in batch["tokens"]]
            

            batch["tokens"] = torch.tensor(batch["tokens"]).to(device)
            batch["date"] = torch.tensor(batch["date"]).to(device)

            output = moe(batch["tokens"], batch["date"],targets=batch["tokens"], get_logits=False, moe=True)
        # output = {"logits": logits, "loss": loss, "aux_losses": aux_losses, "router_logits": router_logits,}
        # loss = criterion(output, batch["tokens"])
            loss = output["loss"]

            loss_sum = loss_sum + loss.item()
            print(loss_sum)
        print("Loss Sum over batches on the Test Set : ", loss_sum)
        print("Loss over test set : ", loss_sum/len(range(0, nb_points, config.batch_size)))
        normalized_loss = loss_sum/len(range(0, nb_points, config.batch_size))
        print("Perplexity on the Test Set : ", math.exp(normalized_loss))

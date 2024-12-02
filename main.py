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
num_proc = min(4, cpu_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = tiktoken.get_encoding("gpt2")

def process_data(example, min_date, max_date):
    text_tokens = tokenizer.encode_ordinary(example["text"])
    text_tokens.append(tokenizer.eot_token)
    date = int(example["date"][:4])
    mask_date = torch.zeros((max_date - min_date + 1)//2)
    mask_date[:(date - min_date + 1)//2] = 1
    max_len = config.sequence_length
    text_tokens = text_tokens[:max_len] 
    text_tokens += [0] * (max_len - len(text_tokens))
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
basic_config = Config(**{
    "moe_num_experts": 10, 
    "moe_softmax_order": "softmax_topk",
    "batch_size": 64,
    "n_embd": 768,
    "moe_routing": None,
    "moe": False
})
print("Dataset loaded")

moe_routings = [None, "standard_gating", "masked"]
gradient_accumulation_steps = 128
for moe_routing in moe_routings:
    config = Config(**{
        "moe_num_experts": (max_date - min_date + 1)/2,
        "moe_softmax_order": "softmax_topk",
        "batch_size": 64,
        "n_embd": 768,
        #"date_list": date_list,
        "moe_routing": moe_routing,
        "moe": moe_routing is not None
    })

    moe = GPTBase(config)
    moe.to(device)

    # print_model_architecture(moe)

    # Training 
    nb_points = 1000000
    print(f"Training on {nb_points} data points")
    print("TODO: Compute number of tokens")
    def count_tokens(dataset):
        return sum(len(tokens) for tokens in dataset["tokens"])
    
    print(f"Number of tokens in train: {count_tokens(fineweb_dataset['train'])}")

    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=nb_points)

    losses = []
    loss = torch.tensor([-1])
    best_loss = 1e9
    print("Starting training")
    for epoch in range(1):
        #for i in tqdm(range(0, len(fineweb_dataset["train"]), config.batch_size), desc=f"Loss = {loss.item()}"):
        for i in tqdm(range(0, nb_points, config.batch_size)):
            batch = fineweb_dataset["train"][i:i+config.batch_size]
            batch["tokens"] = torch.tensor(batch["tokens"]).to(device)
            batch["date"] = torch.tensor(batch["date"]).to(device)

            output = moe(batch["tokens"], batch["date"], 
                        targets=batch["tokens"], get_logits=False, moe=True)
            # output = {"logits": logits, "loss": loss, "aux_losses": aux_losses, "router_logits": router_logits,}
            # loss = criterion(output, batch["tokens"])
            loss = output["loss"]
            loss.backward()
            if (i//config.batch_size) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            del batch["tokens"]
            del batch["date"]
            torch.cuda.empty_cache()
            
            #print(torch.cuda.memory_summary(device=device))

            if i % 1000 == 0:
                print(f"Episode: {i}, Loss: {loss.item()}")
                losses.append(loss.item())
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    save_checkpoint(moe, optimizer, scheduler, i, f"best_model_{str(moe_routing)}.pth")
        
        print(f"Epoch: {epoch}, Loss: {loss.item()}")  

    with open(f"{str(moe_routing)}_losses.json", "w") as f:
        json.dump(losses, f)     

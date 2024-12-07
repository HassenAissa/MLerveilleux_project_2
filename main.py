import datasets
from gpt import GPTBase
import torch
import os
from multiprocessing import cpu_count
import tiktoken
from datasets import load_dataset
from utils import process_data

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
    "batch_size": 64,
    "n_embd": 768,
    "moe_routing": None,
    "moe": False
})


fineweb_dataset, min_date, max_date = get_fineweb_dataset(test = False)

print("Dataset loaded")

# moe_routings = ["standard_gating", "masked"]
moe_routings = ["masked"]
gradient_accumulation_steps = 2
nb_points = 100_000

fineweb_dataset = fineweb_dataset["train"].select(range(nb_points))
fineweb_dataset = fineweb_dataset.map(
    lambda examples: {"tokens": torch.tensor(examples["tokens"]).to(device), "date": torch.tensor(examples["date"]).to(device)},
    batched=True,
    batch_size = 10000
)
for moe_routing in moe_routings:
    config = Config(**{
        "moe_num_experts": (max_date - min_date + 1)//2,
        "moe_softmax_order": "softmax_topk",
        "batch_size": 64,
        "n_embd": 768,
        #"date_list": date_list,
        "moe_routing": moe_routing,
        "moe": moe_routing is not None
    })

    moe = GPTBase(config)
    moe.to(device)

    import torch.nn as nn
    import torch.nn.init as init

    # Define the weight initialization function
    def initialize_weights(module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0, std=0.01)
        elif isinstance(module, nn.Conv2d):
            init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)

    # Instantiate the model
    moe = GPTBase(config)
    moe.to(device)

    # Initialize the weights
    moe.apply(initialize_weights)

    print("Model weights initialized.")


    print_model_architecture(moe)

    # Training 
    print(f"Training on {nb_points} data points")
    lr = 1e-2
    optimizer = torch.optim.AdamW(moe.parameters(), lr=lr, weight_decay = 0.1)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.7, end_factor=0.01, total_iters=nb_points//(config.batch_size * gradient_accumulation_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, total_steps=nb_points//(config.batch_size * gradient_accumulation_steps)+1, 
                                                            pct_start=0.05, anneal_strategy="cos", 
                                                            cycle_momentum=False, div_factor=1e2, final_div_factor=.1)
    #cosine scheduling
    #weight decay 0.1
    #use get X, Y from the github
    #concatenate all the dataset and the select with get X Y
    # 64 * 1024
    losses = []
    loss = torch.tensor([-1])
    best_loss = 1e9
    nb_tokens = 0
    print("Starting training")
    for epoch in range(1):
        for i in tqdm(range(0, nb_points, config.batch_size)):
            batch = fineweb_dataset[i:i+config.batch_size]
            batch["tokens"] = torch.tensor(batch["tokens"]).to(device)
            batch["date"] = torch.tensor(batch["date"]).to(device)
            output = moe(batch["tokens"][:, :-1].clone(), batch["date"], 
                        targets=batch["tokens"][:, 1:].clone(), get_logits=False, moe=config.moe)
            # output = {"logits": logits, "loss": loss, "aux_losses": aux_losses, "router_logits": router_logits,}
            # loss = criterion(output, batch["tokens"])
            loss = output["loss"]
            loss.backward()
            if (i//config.batch_size) % gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(moe.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            del batch["tokens"]
            del batch["date"]
            # torch.cuda.empty_cache()
            
            #print(torch.cuda.memory_summary(device=device))
            nb_tokens += basic_config.sequence_length * config.batch_size
            if (i//config.batch_size) % 10 == 0:
                print(f"Episode: {i}, Loss: {loss.item()}, Tokens seen: {nb_tokens}")
                torch.cuda.empty_cache()
                losses.append(loss.item())
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    save_checkpoint(moe, optimizer, scheduler, i, f"best_model_{str(moe_routing)}_small2.pth")
        
        print(f"Epoch: {epoch}, Loss: {loss.item()}")  

    with open(f"{str(moe_routing)}_losses_small2.json", "w") as f:
        json.dump(losses, f)

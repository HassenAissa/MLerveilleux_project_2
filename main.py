from utils import save_checkpoint
from utils import  get_fineweb_dataset
from multiprocessing import cpu_count
from config import Config
from gpt import GPTBase
from tqdm import tqdm
import datasets
import utils
import torch
import wandb
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.bfloat16)
SEED = 42

# define the types of models to run
moe_routings = [None, "masked", "standard_gating"]
run_names = ["GPT2", "time_dependent", "standard_gating_final"]

#load the dataset
nb_points = 100_000
fineweb_dataset, min_date, max_date = get_fineweb_dataset(nb_points)

gradient_accumulation_steps = 2

# Move the dataset tensors to the device for better performance
utils.move_to_device(fineweb_dataset, device)   

fineweb_dataset_val = fineweb_dataset["test"]
fineweb_dataset = fineweb_dataset["train"]


for moe_routing, run_name in zip(moe_routings, run_names):

    # define the configuration
    config = Config(**{
        "moe_num_experts": (max_date - min_date) // 2 + 1,
        "moe_softmax_order": "softmax_topk",
        "batch_size": 64,
        "n_embd": 768,
        # "date_list": date_list,
        "moe_routing": moe_routing,
        "moe": moe_routing is not None
    })


    # Initialize wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="time_dependant_llm",
        name=run_name,

        # track hyperparameters and run metadata
        config={
            "moe_num_experts": (max_date - min_date) // 2 +1,
            "moe_softmax_order": "softmax_topk",
            "batch_size": 64,
            "n_embd": 768,
            "moe_routing": moe_routing,
            "moe": moe_routing is not None
        }
    )


    # Instantiate the model
    moe = GPTBase(config)
    moe.to(device)

    # Initialize the weights
    moe.apply(utils.initialize_weights)
    moe.train()
    print("Model weights initialized.")

    total_params = sum(p.numel() for p in moe.parameters())
    print(f"Total number of parameters: {total_params}")

    nb_points = len(fineweb_dataset)


    # Training 
    utils.train(
        moe, 
        config, 
        run_name,
        fineweb_dataset, 
        fineweb_dataset_val, 
        device, 
        wandb,
        gradient_accumulation_steps)

    wandb.finish()

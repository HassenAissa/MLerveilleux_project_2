

CHECKPOINTS_PATH = [ "best_model_None2.pth", "best_model_standard_gating2.pth", "best_model_masked2.pth"]
import math 
SEQUENCE_LENGTH = 1024
import datasets
from gpt import GPTBase
import torch
import os
from multiprocessing import cpu_count
import tiktoken
from datasets import load_dataset
from utils import process_data, get_fineweb_dataset
from config import Config
from tqdm import tqdm
import json
import tiktoken
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

print("Starting to load datase")
fineweb_dataset, min_date, max_date = get_fineweb_dataset(test = True)
print("Dataset loaded")
moe_routings = [ None, "standard_gating", "masked"]
fineweb_dataset = fineweb_dataset["test"]


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
            batch["tokens"] = torch.tensor(batch["tokens"]).to(device)
            batch["date"] = torch.tensor(batch["date"]).to(device)
            # print_model_architecture(moe)
            # print(batch["tokens"][:, :-1].shape)
            # print("mask",batch["date"].shape)
            output = moe(batch["tokens"][:, :-1].clone(), batch["date"], 
                        targets=batch["tokens"][:, 1:].clone(), get_logits=False, moe=config.moe)
        # output = {"logits": logits, "loss": loss, "aux_losses": aux_losses, "router_logits": router_logits,}
        # loss = criterion(output, batch["tokens"])
            loss = output["loss"]

            loss_sum = loss_sum + loss.item()
            # print(loss_sum)
        print("Loss Sum over batches on the Test Set : ", loss_sum)
        print("Loss over test set : ", loss_sum/len(range(0, nb_points, config.batch_size)))
        normalized_loss = loss_sum/len(range(0, nb_points, config.batch_size))
        print("Perplexity on the Test Set : ", math.exp(normalized_loss))

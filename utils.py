from datasets import load_dataset, Dataset
from multiprocessing import cpu_count
import torch.nn.init as init
import multiprocessing
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import tiktoken
import random
import torch
import json
import os

SEED = 42
random.seed(SEED)
print("RANDOM SEED SET TO", SEED)

num_proc = max(4, cpu_count())


def process_data(example, min_date, max_date, tokenizer):
    """ tokenize and encode the text and date of the example

    Args:
        example (dict): the example to process
        min_date (int): the minimum date in the dataset
        max_date (int): the maximum date in the dataset
        tokenizer (tiktoken.Tokenizer): the tokenizer to use
    Returns:
        dict: the processed example
    """
    text_tokens = tokenizer.encode_ordinary(example["text"])
    text_tokens.append(tokenizer.eot_token)
    date = (int(example["date"][:4]) - min_date) // 2 + 1
    return {"tokens": text_tokens, "date": date}


def get_fineweb_dataset(nb_points, num_proc=num_proc):
    """Load the fineweb dataset and tokenize it

    Args:
        nb_points (int): the number of points to load
        num_proc (int): the number of processes to use

    Returns:
        tuple: the train and test datasets, the minimum and maximum dates in the dataset
    """
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = load_dataset(path="HuggingFaceFW/fineweb", name="sample-10BT")
    shuffled_dataset = dataset['train'].shuffle(seed=42)
    dataset = shuffled_dataset.select(range(nb_points))
    min_date = int(min(dataset["date"])[:4])
    max_date = int(max(dataset["date"])[:4])
    
    print("Now tokenizing dataset")
    tokenized = dataset.map(
        process_data,
        remove_columns=["text"],
        desc="Tokenizing the splits",
        num_proc=num_proc,
        fn_kwargs={"min_date": min_date, "max_date": max_date, "tokenizer": tokenizer},
    )
    print(tokenized)

    text_years = [[],[],[],[],[],[]]
    with multiprocessing.Pool(128) as pool:
        for text, date in tqdm(zip(tokenized["tokens"], tokenized["date"])):
            text_years[date-1] += text

    new_dataset = {
        "tokens": [],
        "date": []
    }

    with multiprocessing.Pool(128) as pool:
        for j, text in tqdm(enumerate(text_years)):
            for i in range(0, len(text)-1025, 1025):
                new_dataset["tokens"].append(text[i:i+1025])
                new_dataset["date"].append(j+1)
    dataset = Dataset.from_dict(new_dataset)

    split_dataset = dataset.train_test_split(test_size=0.005, seed=SEED, shuffle=True)

    return split_dataset, min_date, max_date



def save_checkpoint(model, opt, scheduler, itr, ckpt_path, **extra_args):
    """Save the model checkpoint

    Args:
        model (nn.Module): the model to save    
        opt (torch.optim.Optimizer): the optimizer to save
        scheduler (torch.optim.lr_scheduler): the scheduler to save
        itr (int): the current iteration
        ckpt_path (str): the path to save the checkpoint to
        **extra_args: any extra arguments to save
    """
    checkpoint = dict({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "itr": itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)


def print_model_architecture(model):
    """ Print the model architecture and the number of parameters in each block
    
    Args:
        model (nn.Module): the model to print the architecture of
    """
    print("Model architecture:\n")
    print(model)

    print("\n\nNumber of parameters in the model:\n")
    total_params = 0
    indent_level = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_params = param.numel()
            block_hierarchy = name.split('.')[:-1]
            while len(block_hierarchy) < indent_level:
                indent_level -= 1
            while len(block_hierarchy) > indent_level:
                print(f"{'    ' * indent_level}Block: {block_hierarchy[indent_level]}")
                indent_level += 1
            print(f"{'    ' * indent_level}Layer: {name} --> Number of parameters: {layer_params}")
            total_params += layer_params
    print(f"\nTotal number of parameters: {total_params}\n")


@torch.no_grad()
def eval(model, val_dataset,config, device='cpu'):
    """Evaluate the model on the validation dataset

    Args:
        model (nn.Module): the model to evaluate
        val_dataset (Dataset): the validation dataset
        config (Config): the configuration of the model
        device (str): the device to use

    Returns:
        float: the validation accuracy
        float: the validation loss
        float: the validation perplexity
    """
    assert model.training == False
    print("Computing validation loss")
    loss_list_val, acc_list = [], []
    nb_points = len(val_dataset)
    for i in tqdm(range(0, nb_points - config.batch_size, config.batch_size)):
        batch = val_dataset[i:i + config.batch_size]
        batch["tokens"] = torch.tensor(batch["tokens"]).to(device)
        batch["date"] = torch.tensor(batch["date"]).to(device)
        x = batch["tokens"][:, :-1].clone()
        y = batch["tokens"][:, 1:].clone()
        outputs = model(x, batch["date"],targets = y, get_logits = False, moe = config.moe)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        #acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())
        del batch["tokens"]
        del batch["date"]
        del x
        del y

    #val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return 0.0, val_loss, val_perplexity


def train(model, config, run_name, fineweb_dataset, fineweb_dataset_val, device, wandb, gradient_accumulation_steps):
    """Train the model

    Args:
        model (nn.Module): the model to train
        config (Config): the configuration of the model
        run_name (str): the name of the run
        fineweb_dataset (Dataset): the training dataset
        fineweb_dataset_val (Dataset): the validation dataset
        nb_points (int): the number of points in the training dataset
        device (str): the device to use
        wandb (wandb): the wandb object to log to
        gradient_accumulation_steps (int): the number of steps to accumulate gradients over

    """
    lr = 1e-3
    nb_points = len(fineweb_dataset)
    total_steps = nb_points // (config.batch_size * gradient_accumulation_steps) + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=lr, 
        total_steps= total_steps,
        pct_start=0.05, 
        anneal_strategy="cos",
        cycle_momentum=False, 
        div_factor=1e1, 
        final_div_factor=.1
    )

    losses = []
    loss = torch.tensor([-1])
    best_loss = 1e9
    nb_tokens = 0
    print("Starting training")
    val_curve_freq = 2000
    for epoch in range(1):
        for i in tqdm(range(0, nb_points - config.batch_size, config.batch_size)):

            batch = fineweb_dataset[i:i + config.batch_size]
            batch["tokens"] = torch.tensor(batch["tokens"]).to(device)
            batch["date"] = torch.tensor(batch["date"]).to(device)

            output = model(batch["tokens"][:, :-1].clone(), batch["date"],
                         targets=batch["tokens"][:, 1:].clone(), get_logits=False, moe=config.moe)

            loss = output["loss"]
            wandb.log({"train/loss": loss.item()}, step=i // config.batch_size)
            loss.backward()

            if (i // config.batch_size) % gradient_accumulation_steps == 0:
                # Update the weights
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            del batch["tokens"]
            del batch["date"]

            nb_tokens += config.sequence_length * config.batch_size

            if (i // config.batch_size) % 10 == 0:
                # Log the loss and the number of tokens seen
                print(f"Episode: {i}, Loss: {loss.item()}, Tokens seen: {nb_tokens}")
                torch.cuda.empty_cache()
                losses.append(loss.item())
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    save_checkpoint(model, optimizer, scheduler, i, f"best_model_{str(run_name)}.pth")
            
            if (i // config.batch_size) % val_curve_freq == 0:
                # Evaluate the model on the validation set
                model.eval()
                val_acc, val_loss, val_perplexity = eval(model, fineweb_dataset_val, config, device=device)
                wandb.log({
                    "val/loss":val_loss,
                    "val/acc":val_acc,
                    "val/perplexity":val_perplexity
                }, step = i//config.batch_size)



        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    with open(f"{str(run_name)}_losses.json", "w") as f:
        json.dump(losses, f)


def initialize_weights(module):
    """Initialize the weights of the model

    Args:
        module (nn.Module): The module to initialize the weights of
    """
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

def move_to_device(dataset, device):
    """Move the dataset tensors to the device
    
    Args:
        dataset (Dataset): the dataset to move to the device
        device (str): the device to move the dataset to
    Returns:
        Dataset: the dataset with the tensors moved to the device
    """
    fineweb_dataset = dataset.map(
        lambda examples: {"tokens": torch.tensor(examples["tokens"]).to(device),
                      "date": torch.tensor(examples["date"]).to(device)},
        batched=True,
        batch_size=10000
    )
    return fineweb_dataset

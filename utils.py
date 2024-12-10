import torch
from multiprocessing import cpu_count
import tiktoken
from datasets import load_dataset
import random
import tqdm
SEED = 42
random.seed(SEED)
print("RANDOM SEED SETTED TO", SEED)

num_proc = max(4, cpu_count())
def process_data2(example, min_date, max_date, tokenizer):
    text_tokens = tokenizer.encode_ordinary(example["text"])
    text_tokens.append(tokenizer.eot_token)
    date = int(example["date"][:4])
    mask_date = torch.zeros((max_date - min_date + 1)//2)
    mask_date[:(date - min_date + 1)//2] = 1
    max_len = 1024
    starting_position = max(0, random.randint(0,max(0,len(text_tokens)-max_len-1)))
    #print("starting pos : ", starting_position)
    #print(len(text_tokens))
    text_tokens = text_tokens[:max_len+1]
    text_tokens += ([tokenizer.eot_token] * (max_len+ 1 - len(text_tokens)))
    # print(len(text_tokens))
    return {"tokens": text_tokens, "date": mask_date}


def get_fineweb_dataset(test, num_proc=num_proc, nb_points = 1_000_000):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = load_dataset(path="HuggingFaceFW/fineweb", name="sample-10BT", cache_dir="../../lcostes/MLerveilleux_project_2/huggingface_cache/datasets")
    print("loaded")
    shuffled_dataset = dataset['train'].shuffle(seed=42)
    dataset = shuffled_dataset.select(range(nb_points))
    split_dataset = dataset.train_test_split(test_size=0.005, seed=SEED, shuffle=True)
    
    min_date = int(min(min(split_dataset["train"]["date"]), min(split_dataset["test"]["date"]))[:4])
    max_date = int(max(max(split_dataset["train"]["date"]), max(split_dataset["test"]["date"]))[:4])
    # if test:
    #     tokenized = split_dataset["test"].map(
    #         process_data,
    #         remove_columns=["text"],
    #         desc="Tokenizing the splits",
    #         num_proc=num_proc,
    #         fn_kwargs={"min_date": min_date, "max_date": max_date, "tokenizer": tokenizer},
    #     )
    # else:
    print("Now tokenizing dataset")
    tokenized = split_dataset.map(
        process_data2,
        remove_columns=["text"],
        desc="Tokenizing the splits",
        num_proc=num_proc,
        fn_kwargs={"min_date": min_date, "max_date": max_date, "tokenizer": tokenizer},
    )
    print("done")
    print(tokenized)
    
    return tokenized, min_date, max_date

def save_checkpoint(model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "itr": itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)


def print_model_architecture(model):
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
    assert model.training == False
    print("Computing validation loss")
    loss_list_val, acc_list = [], []
    nb_points = len(val_dataset)
    for i in tqdm.tqdm(range(0, nb_points - config.batch_size, config.batch_size)):
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

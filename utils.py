import torch


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
import numpy as np
import torch

from .fineweb import get_fineweb_dataset


def get_dataset(dataset):
    """
    Fetch the appropriate dataset based on the provided dataset name.
    This function retrieves the dataset specified by the `dataset` parameter.
    Each dataset has its own corresponding Python file that contains the logic
    for loading and processing the data.

    Args:
        dataset (str): The name of the dataset to fetch. Currently supported values are:
                       - "fineweb": Fetches the FineWeb dataset.
    Returns:
        dict: A dictionary containing two keys, 'train' and 'val', which correspond to the
              tokenized training and validation data, respectively. The values are instances
              of `np.memmap`.
    
    Raises:
        ValueError: If the specified dataset is not supported.
    """
    if dataset == "fineweb":
        return get_fineweb_dataset()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length):
        super().__init__()
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        total_length = len(self.data)
        # chunk the data into sequences of length `sequence_length`
        # NOTE: we discard the last remainding sequence if it's not of length `sequence_length`
        return (total_length - 1) // self.sequence_length

    def __getitem__(self, idx):
        seq_length = self.sequence_length
        idx = idx * seq_length

        x = torch.from_numpy((self.data[idx : idx + seq_length]).astype(np.int64))
        y = torch.from_numpy((self.data[idx + 1 : idx + 1 + seq_length]).astype(np.int64))
        
        return x, y


def get_dataloader(data, sequence_length, batch_size, seed=0, distributed_backend=None):
    """
    Create a DataLoader for the given data.

    If `distributed_backend` is provided and is truly distributed (world size > 1), 
    the DataLoader will be created with a DistributedSampler that splits the data 
    across the processes (in conjunction with DDP). Otherwise, use a RandomSampler 
    with the specified seed.

    Parameters:
        data (Any): The input data to be loaded.
        sequence_length (int): The length of the sequences to be generated from the data.
        batch_size (int): The number of samples per batch to load.
        seed (int, optional): The seed for random number generation. Default is 0.
        distributed_backend (Any, optional): The backend for distributed training. Default is None.
    
    Returns:
        tuple: A tuple containing the DataLoader and the sampler.
    """
    dataset = Dataset(data, sequence_length=sequence_length)
    if distributed_backend and distributed_backend.get_world_size() > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=True,
            seed=seed,
        )
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(
            dataset, replacement=False, generator=g
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=4,
    )
    return loader, sampler
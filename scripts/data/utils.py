from torch.utils.data import random_split, Dataset as TorchDataset
from datasets import Dataset as HFDataset

def train_val_split(
    dataset,
    val_size: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Split either:
    - Hugging Face datasets.Dataset using train_test_split
    - PyTorch torch.utils.data.Dataset using random_split
    """

    # Case 1: Hugging Face Dataset
    if isinstance(dataset, HFDataset):
        split = dataset.train_test_split(
            test_size=val_size,
            shuffle=shuffle,
            seed=seed,
        )
        return split["train"], split["test"]

    # Case 2: PyTorch Dataset
    if isinstance(dataset, TorchDataset):
        n_total = len(dataset)
        n_val = int(round(n_total * val_size))
        n_train = n_total - n_val

        if n_train <= 0 or n_val <= 0:
            raise ValueError(
                f"Invalid split: dataset has {n_total} samples, "
                f"val_size={val_size} gives train={n_train}, val={n_val}"
            )

        import torch

        generator = torch.Generator().manual_seed(seed)

        train_dataset, valid_dataset = random_split(
            dataset,
            lengths=[n_train, n_val],
            generator=generator,
        )

        return train_dataset, valid_dataset

    raise TypeError(
        f"Unsupported dataset type: {type(dataset)}. "
        "Expected a Hugging Face datasets.Dataset or a torch.utils.data.Dataset."
    )
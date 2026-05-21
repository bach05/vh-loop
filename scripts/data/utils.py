from __future__ import annotations

"""Dataset utility helpers."""

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset, random_split


def train_val_split(
    dataset,
    val_size: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
):
    """Split a Hugging Face or PyTorch dataset into train/validation subsets.

    Parameters
    ----------
    dataset:
        Either a ``datasets.Dataset`` or a ``torch.utils.data.Dataset``.
    val_size:
        Fraction of samples assigned to validation. Must be in ``(0, 1)``.
    shuffle:
        Used only for Hugging Face datasets. PyTorch ``random_split`` is random
        by construction through the seeded generator.
    seed:
        Random seed for reproducible splits.
    """
    if not 0 < val_size < 1:
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")

    if isinstance(dataset, HFDataset):
        split = dataset.train_test_split(
            test_size=val_size,
            shuffle=shuffle,
            seed=seed,
        )
        return split["train"], split["test"]

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
        return random_split(
            dataset,
            lengths=[n_train, n_val],
            generator=generator,
        )

    raise TypeError(
        f"Unsupported dataset type: {type(dataset)}. "
        "Expected a Hugging Face datasets.Dataset or a torch.utils.data.Dataset."
    )

"""
Dataset Registry
Registry system for datasets and dataloader builders
"""

import torch
from utils import Registry, build_from_cfg


DATASETS = Registry('datasets')


def build_dataset(split_cfg):
    """
    Build dataset from config.

    Args:
        split_cfg: Dataset split config (e.g., cfg.dataset.train)
        cfg: Global config object (optional)

    Returns:
        Dataset instance

    Example:
        >>> train_dataset = build_dataset(cfg.dataset.train)
    """
    return build_from_cfg(split_cfg, DATASETS)


def build_dataloader(split_cfg, batch_size, num_workers=4, is_train=True):
    """
    Build PyTorch DataLoader from config.

    Args:
        split_cfg: Dataset split config (e.g., cfg.dataset.train)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes (default: 4)
        is_train: Whether this is training data (affects shuffle, drop_last)

    Returns:
        DataLoader instance

    Example:
        >>> train_loader = build_dataloader(cfg.dataset.train, batch_size=16, num_workers=4, is_train=True)
        >>> val_loader = build_dataloader(cfg.dataset.val, batch_size=16, num_workers=4, is_train=False)
    """
    # Build the dataset
    dataset = build_dataset(split_cfg)

    # Shuffle only for training
    shuffle = is_train

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return data_loader

"""
Dataset Registry
Registry system for datasets and dataloader builders
"""

import torch
from utils import Registry, build_from_cfg


DATASETS = Registry('datasets')


def build_dataset(split_cfg, cfg=None):
    """
    Build dataset from config.

    Args:
        split_cfg: Dataset split config (e.g., cfg.dataset.train)
        cfg: Global config object (optional)

    Returns:
        Dataset instance

    Example:
        >>> train_dataset = build_dataset(cfg.dataset.train, cfg)
    """
    default_args = {'cfg': cfg} if cfg is not None else None
    return build_from_cfg(split_cfg, DATASETS, default_args=default_args)


def build_dataloader(split_cfg, cfg, is_train=True):
    """
    Build PyTorch DataLoader from config.

    Args:
        split_cfg: Dataset split config (e.g., cfg.dataset.train)
        cfg: Global config object
        is_train: Whether this is training data (affects shuffle)

    Returns:
        DataLoader instance

    Example:
        >>> train_loader = build_dataloader(cfg.dataset.train, cfg, is_train=True)
        >>> val_loader = build_dataloader(cfg.dataset.val, cfg, is_train=False)
    """
    # Build the dataset
    dataset = build_dataset(split_cfg, cfg)

    # Shuffle only for training
    shuffle = is_train

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False
    )

    return data_loader

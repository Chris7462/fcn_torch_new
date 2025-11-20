"""
Transform Registry
Registry system for data transforms
"""

from utils import Registry, build_from_cfg


TRANSFORMS = Registry('transforms')


def build_transform(cfg):
    """
    Build a single transform from config.

    Args:
        cfg: Transform config dict with 'type' key

    Returns:
        Transform instance

    Example:
        >>> cfg = dict(type='Resize', height=360, width=480)
        >>> transform = build_transform(cfg)
        >>> sample = transform({'img': img, 'mask': mask})
    """
    return build_from_cfg(cfg, TRANSFORMS)

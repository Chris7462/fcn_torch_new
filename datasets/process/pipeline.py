"""
Pipeline for composing multiple transforms
"""


class Pipeline:
    """
    Process pipeline that applies a sequence of transforms.

    Args:
        transforms: List of transform config dicts

    Example:
        >>> pipeline = Pipeline([
        >>>     dict(type='Resize', height=360, width=480),
        >>>     dict(type='Normalize', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        >>>     dict(type='ToTensor')
        >>> ])
        >>> sample = {'img': img, 'mask': mask}
        >>> sample = pipeline(sample)
    """

    def __init__(self, transforms):
        """
        Initialize pipeline with a list of transform configs.

        Args:
            transforms: List of dicts, each with 'type' key and transform params
        """
        from .registry import build_transform

        self.transforms = []

        if transforms is None:
            transforms = []

        for transform_cfg in transforms:
            transform = build_transform(transform_cfg)
            self.transforms.append(transform)

    def __call__(self, sample):
        """
        Apply all transforms sequentially to the sample.

        Args:
            sample: Dict with at least 'img' key, optionally 'mask', 'meta', etc.

        Returns:
            Transformed sample dict
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self):
        """String representation of the pipeline"""
        format_str = self.__class__.__name__ + '('
        for t in self.transforms:
            format_str += f'\n    {t}'
        format_str += '\n)'
        return format_str

"""
Minimal abstract base dataset for all datasets.
"""

from torch.utils.data import Dataset
from .registry import DATASETS


@DATASETS.register_module
class BaseDataset(Dataset):
    """
    Abstract base class for all datasets.

    Subclasses must implement:
        - load_annotations(): Load dataset-specific annotations
        - prepare_data(idx): Prepare raw data before transforms

    Args:
        data_root: Root directory of dataset
        split: Split name ('train', 'val', 'test')
        processes: Transform pipeline config (list of dicts)

    Example:
        >>> class MyDataset(BaseDataset):
        >>>     def load_annotations(self):
        >>>         # Load your annotations
        >>>         self.data_infos = [...]
        >>>
        >>>     def prepare_data(self, idx):
        >>>         # Load image and mask
        >>>         return {'img': img, 'mask': mask}
    """

    def __init__(self, data_root, split, processes=None):
        """
        Initialize base dataset.

        Args:
            data_root: Root directory of dataset
            split: Split name (used to determine if training)
            processes: List of transform config dicts
        """
        self.data_root = data_root
        self.split = split
        self.is_training = 'train' in split

        # Load dataset-specific annotations (subclass implements this)
        self.data_infos = []
        self.load_annotations()

        # Build transform pipeline from config
        from .process import Pipeline
        self.pipeline = Pipeline(processes)

    def load_annotations(self):
        """
        Load dataset annotations.
        Must be implemented by subclasses.

        This method should populate self.data_infos with a list of dicts,
        where each dict contains information needed to load one sample.

        Example:
            self.data_infos = [
                {'filename': 'image1.png', 'label': 0},
                {'filename': 'image2.png', 'label': 1},
            ]

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement load_annotations()'
        )

    def prepare_data(self, idx):
        """
        Prepare raw data before applying transforms.
        Must be implemented by subclasses.

        This method should load the raw image (and mask if training) and
        return a sample dict with at least an 'img' key.

        Args:
            idx: Index of the sample to load

        Returns:
            dict: Sample dict with at least 'img' key, optionally 'mask', 'meta', etc.

        Example:
            return {
                'img': image,           # Required: numpy array (H, W, C)
                'mask': mask,           # Optional: numpy array (H, W)
                'filename': 'xxx.png',  # Optional: metadata
            }

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f'{self.__class__.__name__} must implement prepare_data(idx)'
        )

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.data_infos)

    def __getitem__(self, idx):
        """
        Load one sample and apply transforms.

        Args:
            idx: Index of the sample to load

        Returns:
            dict: Transformed sample with 'img', 'mask' (if training), etc.
        """
        # Get raw data (subclass implements this)
        sample = self.prepare_data(idx)

        # Apply transform pipeline
        sample = self.pipeline(sample)

        return sample

    def __repr__(self):
        """String representation of the dataset"""
        return (f'{self.__class__.__name__}('
                f'split={self.split}, '
                f'num_samples={len(self)}, '
                f'is_training={self.is_training})')

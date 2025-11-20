"""
CamVid Dataset for Semantic Segmentation
11 classes grouped from original 32 classes
"""

import os
import json
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class CamVid(BaseDataset):
    """
    CamVid Dataset - 11 classes for semantic segmentation.

    Args:
        img_dir: Directory containing raw images (701_StillsRaw_full)
        label_dir: Directory containing label images (LabeledApproved_full)
        split_file: Path to split file (train.txt, val.txt, test.txt)
        dataset_info_path: Path to dataset_info.json
        processes: Transform pipeline config (list of dicts)
        cfg: Global config object

    Example:
        >>> dataset = CamVid(
        >>>     img_dir='./data/CamVid/701_StillsRaw_full',
        >>>     label_dir='./data/CamVid/LabeledApproved_full',
        >>>     split_file='./data/CamVid/splits/train.txt',
        >>>     dataset_info_path='./data/CamVid/splits/dataset_info.json',
        >>>     processes=[
        >>>         dict(type='Resize', height=360, width=480),
        >>>         dict(type='Normalize', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        >>>         dict(type='ToTensor')
        >>>     ],
        >>>     cfg=cfg
        >>> )
    """

    def __init__(self, img_dir, label_dir, split_file, dataset_info_path,
                 processes=None, cfg=None):
        """
        Initialize CamVid dataset.

        Args:
            img_dir: Directory containing raw images
            label_dir: Directory containing label images
            split_file: Path to split file
            dataset_info_path: Path to dataset_info.json
            processes: Transform pipeline config
            cfg: Global config object
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.split_file = split_file

        # Load CamVid-specific metadata
        self.load_dataset_info(dataset_info_path)

        # Call parent init (will call load_annotations and build pipeline)
        super().__init__(
            data_root=img_dir,
            split=os.path.basename(split_file),
            processes=processes,
            cfg=cfg
        )

    def load_dataset_info(self, json_path):
        """
        Load CamVid metadata from JSON file.

        This includes:
            - Number of classes
            - Class names
            - Ignore index
            - Mean and std for normalization
            - Color to class mapping

        Args:
            json_path: Path to dataset_info.json
        """
        with open(json_path, 'r') as f:
            info = json.load(f)

        self.num_classes = info['num_classes']
        self.class_names = info['class_names']
        self.ignore_index = info['ignore_index']
        self.mean = info.get('mean', [0.485, 0.456, 0.406])
        self.std = info.get('std', [0.229, 0.224, 0.225])

        # Build color to class mapping for RGB masks
        self.color_to_class = {}
        for color_str, class_idx in info['color_to_class'].items():
            # Parse "(r, g, b)" string to tuple
            r, g, b = eval(color_str)
            self.color_to_class[(r, g, b)] = class_idx

    def load_annotations(self):
        """
        Load CamVid file list.

        Reads the split file (train.txt, val.txt, test.txt) where each line
        contains a filename in format: city/basename

        Example line: aachen/aachen_000000_000019
        """
        with open(self.split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data_infos.append({'filename': line})

    def rgb_to_mask(self, mask_rgb):
        """
        Convert RGB mask to class indices.

        CamVid uses RGB color-coded masks where each color represents a class.
        This method converts the RGB mask to a single-channel mask with class indices.

        Args:
            mask_rgb: RGB mask as numpy array (H, W, 3)

        Returns:
            mask: Class index mask as numpy array (H, W)
        """
        h, w = mask_rgb.shape[:2]
        mask = np.full((h, w), self.ignore_index, dtype=np.int64)

        # Map each RGB color to its class index
        for (r, g, b), class_idx in self.color_to_class.items():
            matches = (mask_rgb[:, :, 0] == r) & \
                      (mask_rgb[:, :, 1] == g) & \
                      (mask_rgb[:, :, 2] == b)
            mask[matches] = class_idx

        return mask

    def prepare_data(self, idx):
        """
        Load raw image and label for CamVid.

        Args:
            idx: Index of the sample to load

        Returns:
            dict: Sample dict with 'img', 'mask' (if training), 'filename'
        """
        data_info = self.data_infos[idx]
        filename = data_info['filename']

        # Load raw image
        img_path = os.path.join(self.img_dir, filename)
        image = np.array(Image.open(img_path).convert('RGB'))

        # Prepare sample dict
        sample = {
            'img': image,
            'filename': filename
        }

        # Load mask if training
        if self.is_training:
            # Label filename: replace .png with _L.png
            label_filename = filename[:-4] + '_L.png'
            label_path = os.path.join(self.label_dir, label_filename)

            # Load RGB mask and convert to class indices
            label_rgb = np.array(Image.open(label_path).convert('RGB'))
            mask = self.rgb_to_mask(label_rgb)
            sample['mask'] = mask

        return sample

    def __repr__(self):
        """String representation of CamVid dataset"""
        return (f'{self.__class__.__name__}('
                f'split={self.split}, '
                f'num_samples={len(self)}, '
                f'num_classes={self.num_classes}, '
                f'is_training={self.is_training})')

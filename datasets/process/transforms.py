"""
Transform classes for data augmentation.
Each transform operates on a sample dict: {'img': ..., 'mask': ..., 'meta': ...}
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from .registry import TRANSFORMS


@TRANSFORMS.register_module
class Resize:
    """
    Resize image and mask to target size.

    Args:
        height: Target height
        width: Target width
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transform = A.Resize(height=height, width=width)

    def __call__(self, sample):
        """Apply resize to image and mask (if present)"""
        if 'mask' in sample:
            transformed = self.transform(image=sample['img'], mask=sample['mask'])
            sample['img'] = transformed['image']
            sample['mask'] = transformed['mask']
        else:
            transformed = self.transform(image=sample['img'])
            sample['img'] = transformed['image']
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(height={self.height}, width={self.width})'


@TRANSFORMS.register_module
class CenterCrop:
    """
    Center crop image and mask to target size.

    Args:
        height: Target height
        width: Target width
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transform = A.CenterCrop(height=height, width=width)

    def __call__(self, sample):
        """Apply center crop to image and mask (if present)"""
        if 'mask' in sample:
            transformed = self.transform(image=sample['img'], mask=sample['mask'])
            sample['img'] = transformed['image']
            sample['mask'] = transformed['mask']
        else:
            transformed = self.transform(image=sample['img'])
            sample['img'] = transformed['image']
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(height={self.height}, width={self.width})'


@TRANSFORMS.register_module
class RandomCrop:
    """
    Random crop image and mask to target size.

    Args:
        height: Target height
        width: Target width
    """

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transform = A.RandomCrop(height=height, width=width)

    def __call__(self, sample):
        """Apply random crop to image and mask (if present)"""
        if 'mask' in sample:
            transformed = self.transform(image=sample['img'], mask=sample['mask'])
            sample['img'] = transformed['image']
            sample['mask'] = transformed['mask']
        else:
            transformed = self.transform(image=sample['img'])
            sample['img'] = transformed['image']
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(height={self.height}, width={self.width})'


@TRANSFORMS.register_module
class HorizontalFlip:
    """
    Random horizontal flip.

    Args:
        p: Probability of applying the flip (default: 0.5)
    """

    def __init__(self, p=0.5):
        self.p = p
        self.transform = A.HorizontalFlip(p=p)

    def __call__(self, sample):
        """Apply horizontal flip to image and mask (if present)"""
        if 'mask' in sample:
            transformed = self.transform(image=sample['img'], mask=sample['mask'])
            sample['img'] = transformed['image']
            sample['mask'] = transformed['mask']
        else:
            transformed = self.transform(image=sample['img'])
            sample['img'] = transformed['image']
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


@TRANSFORMS.register_module
class VerticalFlip:
    """
    Random vertical flip.

    Args:
        p: Probability of applying the flip (default: 0.5)
    """

    def __init__(self, p=0.5):
        self.p = p
        self.transform = A.VerticalFlip(p=p)

    def __call__(self, sample):
        """Apply vertical flip to image and mask (if present)"""
        if 'mask' in sample:
            transformed = self.transform(image=sample['img'], mask=sample['mask'])
            sample['img'] = transformed['image']
            sample['mask'] = transformed['mask']
        else:
            transformed = self.transform(image=sample['img'])
            sample['img'] = transformed['image']
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


@TRANSFORMS.register_module
class ColorJitter:
    """
    Random color jittering (only applied to image, not mask).

    Args:
        brightness: Brightness adjustment range (default: 0.2)
        contrast: Contrast adjustment range (default: 0.2)
        saturation: Saturation adjustment range (default: 0.2)
        hue: Hue adjustment range (default: 0.1)
        p: Probability of applying the transform (default: 0.5)
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p
        self.transform = A.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            p=p
        )

    def __call__(self, sample):
        """Apply color jitter to image only (not mask)"""
        sample['img'] = self.transform(image=sample['img'])['image']
        return sample

    def __repr__(self):
        return (f'{self.__class__.__name__}(brightness={self.brightness}, '
                f'contrast={self.contrast}, saturation={self.saturation}, '
                f'hue={self.hue}, p={self.p})')


@TRANSFORMS.register_module
class RandomRotate:
    """
    Random rotation.

    Args:
        limit: Rotation range in degrees (default: 10)
        p: Probability of applying the rotation (default: 0.5)
    """

    def __init__(self, limit=10, p=0.5):
        self.limit = limit
        self.p = p
        self.transform = A.Rotate(limit=limit, p=p)

    def __call__(self, sample):
        """Apply rotation to image and mask (if present)"""
        if 'mask' in sample:
            transformed = self.transform(image=sample['img'], mask=sample['mask'])
            sample['img'] = transformed['image']
            sample['mask'] = transformed['mask']
        else:
            transformed = self.transform(image=sample['img'])
            sample['img'] = transformed['image']
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(limit={self.limit}, p={self.p})'


@TRANSFORMS.register_module
class Normalize:
    """
    Normalize image with mean and std (only applied to image, not mask).

    Args:
        mean: Mean values for normalization [R, G, B]
        std: Std values for normalization [R, G, B]
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.transform = A.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        """Apply normalization to image only (not mask)"""
        sample['img'] = self.transform(image=sample['img'])['image']
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


@TRANSFORMS.register_module
class ToTensor:
    """
    Convert numpy arrays to PyTorch tensors.
    - Image: (H, W, C) -> (C, H, W)
    - Mask: (H, W) -> (H, W) as Long tensor
    """

    def __init__(self):
        self.transform = ToTensorV2()

    def __call__(self, sample):
        """Convert image and mask (if present) to tensors"""
        if 'mask' in sample:
            transformed = self.transform(image=sample['img'], mask=sample['mask'])
            sample['img'] = transformed['image']
            sample['mask'] = transformed['mask'].long()
        else:
            transformed = self.transform(image=sample['img'])
            sample['img'] = transformed['image']
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@TRANSFORMS.register_module
class RandomBrightnessContrast:
    """
    Random brightness and contrast adjustment.

    Args:
        brightness_limit: Brightness adjustment range (default: 0.2)
        contrast_limit: Contrast adjustment range (default: 0.2)
        p: Probability of applying the transform (default: 0.5)
    """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, p=0.5):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p
        self.transform = A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=p
        )

    def __call__(self, sample):
        """Apply brightness/contrast adjustment to image only"""
        sample['img'] = self.transform(image=sample['img'])['image']
        return sample

    def __repr__(self):
        return (f'{self.__class__.__name__}(brightness_limit={self.brightness_limit}, '
                f'contrast_limit={self.contrast_limit}, p={self.p})')


@TRANSFORMS.register_module
class GaussianBlur:
    """
    Apply Gaussian blur.

    Args:
        blur_limit: Maximum kernel size (default: 7)
        p: Probability of applying the transform (default: 0.5)
    """

    def __init__(self, blur_limit=7, p=0.5):
        self.blur_limit = blur_limit
        self.p = p
        self.transform = A.GaussianBlur(blur_limit=blur_limit, p=p)

    def __call__(self, sample):
        """Apply Gaussian blur to image only"""
        sample['img'] = self.transform(image=sample['img'])['image']
        return sample

    def __repr__(self):
        return f'{self.__class__.__name__}(blur_limit={self.blur_limit}, p={self.p})'

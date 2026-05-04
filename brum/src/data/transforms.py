"""Silver layer: albumentations augmentation pipelines.

Targets albumentations==2.0.8.

Key API changes in albumentations 2.x:
- GaussNoise uses std_range=(low, high) not var_limit
- ToTensorV2 is in albumentations.pytorch
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization stats
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(chip_size: int = 512) -> A.Compose:
    """Return the training augmentation pipeline.

    Applies spatial and photometric augmentations, then ImageNet normalization.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.4
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(10 / 255, 50 / 255), p=0.3),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )


def get_val_transforms(chip_size: int = 512) -> A.Compose:
    """Return the validation/test augmentation pipeline (no augmentation)."""
    return A.Compose(
        [
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )

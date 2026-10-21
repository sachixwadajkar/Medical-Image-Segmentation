"""PyTorch Dataset and DataLoader for cell nuclei segmentation in histopathology."""

import os
import glob
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class NucleiDataset(Dataset):
    """Histopathology image and nuclei segmentation mask dataset.

    Args:
        images_dir: Path to directory containing image patches.
        masks_dir: Path to directory containing corresponding mask patches.
        augment: Whether to apply random spatial augmentations (flips, 90-deg rotations).
        transform: Optional additional transform callable.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment
        self.transform = transform

        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.samples: List[Tuple[str, str]] = []

        for img_p in image_paths:
            base = os.path.basename(img_p)
            msk_p = os.path.join(masks_dir, base)
            if os.path.exists(msk_p):
                self.samples.append((img_p, msk_p))
            else:
                raise FileNotFoundError(f"Missing corresponding mask for image: {img_p}")

        if len(self.samples) == 0:
            raise ValueError(f"No image-mask pairs found in {images_dir} and {masks_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_spatial_augmentations(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply joint random dihedral group augmentations (8-fold symmetry).

        Histopathology tissue cuts have rotational invariance (no natural 'up' or 'down').
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        # Random 90-degree rotations (0, 90, 180, 270)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, msk_path = self.samples[idx]

        # Read image (RGB) and mask (Grayscale)
        img = Image.open(img_path).convert("RGB")
        msk = Image.open(msk_path).convert("L")

        img_arr = np.array(img, dtype=np.float32) / 255.0  # Normalized [0, 1]
        msk_arr = (np.array(msk, dtype=np.float32) > 127.5).astype(np.float32)  # Binary {0, 1}

        if self.augment:
            img_arr, msk_arr = self._apply_spatial_augmentations(img_arr, msk_arr)

        # Convert to Channel-First PyTorch Tensors: (C, H, W)
        # Image: (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).contiguous()
        # Mask: (H, W) -> (1, H, W)
        msk_tensor = torch.from_numpy(msk_arr).unsqueeze(0).contiguous()

        if self.transform is not None:
            img_tensor, msk_tensor = self.transform(img_tensor, msk_tensor)

        return img_tensor, msk_tensor


def create_dataloaders(
    data_dir: str = "data",
    batch_size: int = 8,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Construct PyTorch DataLoaders for train, validation, and optional test sets.

    Args:
        data_dir: Root dataset folder containing train/ and val/ subdirectories.
        batch_size: Number of patches per batch.
        num_workers: Worker subprocesses for data loading (0 recommended on Windows).

    Returns:
        train_loader, val_loader, test_loader (or None)
    """
    train_dataset = NucleiDataset(
        images_dir=os.path.join(data_dir, "train", "images"),
        masks_dir=os.path.join(data_dir, "train", "masks"),
        augment=True,
    )

    val_dataset = NucleiDataset(
        images_dir=os.path.join(data_dir, "val", "images"),
        masks_dir=os.path.join(data_dir, "val", "masks"),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = None
    test_img_dir = os.path.join(data_dir, "test", "images")
    test_msk_dir = os.path.join(data_dir, "test", "masks")
    if os.path.exists(test_img_dir) and os.path.exists(test_msk_dir):
        test_dataset = NucleiDataset(images_dir=test_img_dir, masks_dir=test_msk_dir, augment=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader, test_loader

"""Unit tests for dataset loading, tensor conversions, and augmentations."""

import os
import numpy as np
import pytest
import torch
from src.dataset import NucleiDataset, create_dataloaders


def test_dataset_loading():
    train_img_dir = "data/train/images"
    train_msk_dir = "data/train/masks"

    assert os.path.exists(train_img_dir), "data/train/images does not exist"
    assert os.path.exists(train_msk_dir), "data/train/masks does not exist"

    dataset = NucleiDataset(train_img_dir, train_msk_dir, augment=False)
    assert len(dataset) > 0, "Dataset is empty"

    img_tensor, msk_tensor = dataset[0]

    # Verify tensor shapes and dtypes
    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(msk_tensor, torch.Tensor)
    assert img_tensor.ndim == 3 and img_tensor.shape[0] == 3
    assert msk_tensor.ndim == 3 and msk_tensor.shape[0] == 1
    assert img_tensor.dtype == torch.float32
    assert msk_tensor.dtype == torch.float32

    # Verify normalization bounds
    assert img_tensor.min() >= 0.0
    assert img_tensor.max() <= 1.0

    # Verify binary mask values
    unique_mask_vals = torch.unique(msk_tensor).tolist()
    for val in unique_mask_vals:
        assert val in (0.0, 1.0)


def test_dataloader_batch_dimensions():
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="data",
        batch_size=4,
        num_workers=0,
    )

    batch_imgs, batch_msks = next(iter(train_loader))
    assert batch_imgs.shape == (4, 3, 256, 256)
    assert batch_msks.shape == (4, 1, 256, 256)

    val_imgs, val_msks = next(iter(val_loader))
    assert val_imgs.shape == (4, 3, 256, 256)
    assert val_msks.shape == (4, 1, 256, 256)


def test_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        NucleiDataset("non_existent_folder_images", "data/train/masks")


def test_xml_to_mask_parsing():
    import glob
    from src.prepare_data import parse_monuseg_xml

    xml_files = glob.glob("data/raw/**/*.xml", recursive=True)
    assert len(xml_files) > 0, "No raw XML files found to test"

    sample_xml = xml_files[0]
    mask, num_nuclei = parse_monuseg_xml(sample_xml, target_shape=(1000, 1000))

    assert mask.shape == (1000, 1000)
    assert mask.dtype == np.uint8
    assert num_nuclei > 0
    unique_vals = np.unique(mask).tolist()
    for v in unique_vals:
        assert v in (0, 255)


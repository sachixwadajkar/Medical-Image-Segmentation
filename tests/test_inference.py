"""Unit tests for sliding-window tiled inference, probability blending, and nuclei quantification."""

import os
import numpy as np
import pytest
import torch

from src.inference import (
    create_2d_window,
    get_sliding_window_coords,
    tiled_inference,
    quantify_nuclei,
)
from src.model import UNet
from src.predict import load_model_from_checkpoint


def test_2d_window_creation():
    win = create_2d_window(tile_size=256, window_type="hann")
    assert win.shape == (256, 256)
    assert win.min() >= 0.0
    assert np.isclose(win.max(), 1.0, atol=1e-2)
    # Center should have highest weight
    assert win[128, 128] > win[0, 0]


def test_sliding_window_coordinates_coverage():
    img_h, img_w = 1000, 1000
    tile_size = 256
    overlap = 64

    coords = get_sliding_window_coords(img_h, img_w, tile_size=tile_size, overlap=overlap)
    assert len(coords) > 0

    # Ensure every single pixel in 1000x1000 is covered by at least one tile
    coverage = np.zeros((img_h, img_w), dtype=np.int32)
    for y, x in coords:
        assert y >= 0 and x >= 0
        assert y + tile_size <= img_h
        assert x + tile_size <= img_w
        coverage[y : y + tile_size, x : x + tile_size] += 1

    assert (coverage >= 1).all(), "Uncovered pixels found in canvas!"


def test_tiled_inference_output_shape():
    model = UNet(in_channels=3, out_channels=1, init_features=16, bilinear=True)
    dummy_image = np.random.randint(0, 256, (600, 600, 3), dtype=np.uint8)

    result = tiled_inference(
        model=model,
        image=dummy_image,
        tile_size=256,
        overlap=64,
        batch_size=2,
        blend=True,
        threshold=0.5,
        device=torch.device("cpu"),
    )

    assert result["probability_map"].shape == (600, 600)
    assert result["binary_mask"].shape == (600, 600)
    assert result["tile_count"] > 1
    assert result["inference_time_seconds"] > 0.0


def test_nuclei_quantification():
    mask = np.zeros((200, 200), dtype=np.float32)
    # Place two synthetic circular nuclei
    mask[20:40, 20:40] = 1.0  # Area 400 px
    mask[100:130, 100:130] = 1.0  # Area 900 px

    quant = quantify_nuclei(mask, microns_per_pixel=0.25, min_area_pixels=15)
    summary = quant["summary"]
    df = quant["components_df"]

    assert summary["total_detected_nuclei"] == 2
    assert len(df) == 2
    assert "area_pixels" in df.columns
    assert "area_microns2" in df.columns


def test_checkpoint_loading():
    ckpt_path = "checkpoints/best_model.pth"
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    model = load_model_from_checkpoint(ckpt_path, torch.device("cpu"))
    assert isinstance(model, UNet)
    assert model.training is False

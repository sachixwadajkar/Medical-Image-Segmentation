"""Unit tests for U-Net model architecture, forward pass, and gradient backpropagation."""

import pytest
import torch
from src.model import UNet, DoubleConv, Down, Up


def test_double_conv_shape():
    block = DoubleConv(in_channels=3, out_channels=16)
    x = torch.randn(2, 3, 64, 64)
    out = block(x)
    assert out.shape == (2, 16, 64, 64)


def test_down_block_shape():
    down = Down(in_channels=16, out_channels=32)
    x = torch.randn(2, 16, 64, 64)
    out = down(x)
    assert out.shape == (2, 32, 32, 32)


def test_up_block_shape():
    up = Up(in_channels=32, out_channels=16)
    x1 = torch.randn(2, 32, 32, 32)
    x2 = torch.randn(2, 16, 64, 64)
    out = up(x1, x2)
    assert out.shape == (2, 16, 64, 64)


def test_unet_forward_standard_input():
    model = UNet(in_channels=3, out_channels=1, init_features=16)
    x = torch.randn(1, 3, 256, 256)
    logits = model(x)
    assert logits.shape == (1, 1, 256, 256)


def test_unet_predict_probability_bounds():
    model = UNet(in_channels=3, out_channels=1, init_features=16)
    x = torch.randn(2, 3, 128, 128)
    prob = model.predict_probability(x)
    assert prob.shape == (2, 1, 128, 128)
    assert (prob >= 0.0).all()
    assert (prob <= 1.0).all()


def test_unet_predict_mask_binary():
    model = UNet(in_channels=3, out_channels=1, init_features=16)
    x = torch.randn(1, 3, 128, 128)
    mask = model.predict_mask(x, threshold=0.5)
    assert mask.shape == (1, 1, 128, 128)
    unique_vals = torch.unique(mask).tolist()
    for v in unique_vals:
        assert v in (0.0, 1.0)


def test_unet_backward_gradient_flow():
    model = UNet(in_channels=3, out_channels=1, init_features=16)
    x = torch.randn(2, 3, 128, 128)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()

    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    # Verify that gradients exist and are finite for all model parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

"""Unit tests for segmentation metrics, confusion matrix, and loss functions."""

import numpy as np
import pytest
import torch

from src.evaluate import (
    calculate_confusion_matrix,
    calculate_metrics,
    DiceLoss,
    BCEDiceLoss,
)


def test_perfect_prediction():
    gt = np.zeros((64, 64), dtype=np.float32)
    gt[10:30, 10:30] = 1.0
    pred = gt.copy()

    metrics = calculate_metrics(pred, gt)
    assert np.isclose(metrics["dice"], 1.0)
    assert np.isclose(metrics["iou"], 1.0)
    assert np.isclose(metrics["precision"], 1.0)
    assert np.isclose(metrics["recall"], 1.0)
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0


def test_disjoint_prediction():
    gt = np.zeros((64, 64), dtype=np.float32)
    gt[0:20, 0:20] = 1.0
    pred = np.zeros((64, 64), dtype=np.float32)
    pred[30:50, 30:50] = 1.0

    metrics = calculate_metrics(pred, gt)
    assert np.isclose(metrics["dice"], 0.0, atol=1e-4)
    assert np.isclose(metrics["iou"], 0.0, atol=1e-4)
    assert np.isclose(metrics["precision"], 0.0, atol=1e-4)
    assert np.isclose(metrics["recall"], 0.0, atol=1e-4)
    assert metrics["tp"] == 0


def test_empty_masks_handling():
    # Both ground truth and prediction are completely empty
    gt = np.zeros((32, 32), dtype=np.float32)
    pred = np.zeros((32, 32), dtype=np.float32)

    metrics = calculate_metrics(pred, gt)
    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0

    # Ground truth is empty, but prediction is non-empty (all false positives)
    pred[5:10, 5:10] = 1.0
    metrics_fp = calculate_metrics(pred, gt)
    assert np.isclose(metrics_fp["dice"], 0.0, atol=1e-4)
    assert np.isclose(metrics_fp["iou"], 0.0, atol=1e-4)
    assert np.isclose(metrics_fp["precision"], 0.0, atol=1e-4)


def test_loss_gradients():
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    loss = criterion(logits, targets)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()
    assert torch.sum(torch.abs(logits.grad)).item() > 0.0

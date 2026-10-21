"""Evaluation metrics and loss functions for medical image segmentation."""

from typing import Dict, List, Optional, Tuple, Union
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation with smooth numerator and denominator.

    Dice = 2 * |P ∩ G| / (|P| + |G|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:

        logits: Tensor of shape (B, 1, H, W) or (B, H, W) with unnormalized predictions.
        targets: Binary ground truth tensor of shape (B, 1, H, W) or (B, H, W).
        """
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy with Logits and Soft Dice Loss.

    Loss = bce_weight * BCE + dice_weight * Dice
    Balances local pixel-level classification (BCE) with global spatial region overlap (Dice).
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_confusion_matrix(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
) -> Dict[str, int]:
    """Calculate true positive, false positive, false negative, and true negative pixel counts.

    Args:
        pred: Binary array/tensor {0, 1}
        gt: Binary array/tensor {0, 1}
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    p = (pred > 0.5).astype(bool)
    g = (gt > 0.5).astype(bool)

    tp = int(np.sum(p & g))
    fp = int(np.sum(p & (~g)))
    fn = int(np.sum((~p) & g))
    tn = int(np.sum((~p) & (~g)))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def calculate_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Calculate Dice, IoU (Jaccard), Precision, Recall, and Accuracy.

    Explicitly handles empty ground truth and prediction masks without division by zero.
    """
    cm = calculate_confusion_matrix(pred, gt)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

    # Special case: both mask and prediction are completely empty (all background)
    if (tp + fp + fn) == 0:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "accuracy": 1.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    total_pixels = tp + fp + fn + tn
    accuracy = (tp + tn) / (total_pixels + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@torch.no_grad()
def evaluate_dataloader(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate a trained model over a DataLoader.

    Returns averaged aggregate loss and segmentation metrics across all batches.
    """
    model.eval()
    total_loss = 0.0
    accumulated_tp = 0
    accumulated_fp = 0
    accumulated_fn = 0
    accumulated_tn = 0
    batch_count = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        if criterion is not None:
            loss = criterion(logits, targets)
            total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        cm = calculate_confusion_matrix(preds, targets)
        accumulated_tp += cm["tp"]
        accumulated_fp += cm["fp"]
        accumulated_fn += cm["fn"]
        accumulated_tn += cm["tn"]
        batch_count += 1

    avg_loss = (total_loss / batch_count) if batch_count > 0 else 0.0

    # Aggregate global metrics
    eps = 1e-7
    tp = accumulated_tp
    fp = accumulated_fp
    fn = accumulated_fn
    tn = accumulated_tn

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "loss": float(avg_loss),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_dataset_and_save_csv(
    model: nn.Module,
    images_dir: str,
    masks_dir: str,
    output_csv: str,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate individual patch/image samples and save detailed per-sample metrics to CSV."""
    import glob

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    records = []

    model.eval()
    for img_p in tqdm(image_paths, desc="Evaluating samples"):
        base_name = os.path.basename(img_p)
        msk_p = os.path.join(masks_dir, base_name)
        if not os.path.exists(msk_p):
            continue

        pil_img = Image.open(img_p).convert("RGB")
        pil_msk = Image.open(msk_p).convert("L")

        img_arr = np.array(pil_img, dtype=np.float32) / 255.0
        gt_arr = (np.array(pil_msk, dtype=np.float32) > 127.5).astype(np.float32)

        tensor_img = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = model.predict_probability(tensor_img).squeeze().cpu().numpy()
            pred = (prob > threshold).astype(np.float32)

        metrics = calculate_metrics(pred, gt_arr)
        record = {
            "image": base_name,
            "dice": round(metrics["dice"], 4),
            "iou": round(metrics["iou"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "gt_nuclei_pixels": int(np.sum(gt_arr)),
            "pred_nuclei_pixels": int(np.sum(pred)),
        }
        records.append(record)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved evaluation metrics to: {output_csv}")

    # Print summary statistics
    print("\n" + "=" * 55)
    print(f" EVALUATION SUMMARY ({len(df)} samples)")
    print("=" * 55)
    print(f"  Dice Coefficient: {df['dice'].mean():.4f} +/- {df['dice'].std():.4f}")
    print(f"  IoU (Jaccard):    {df['iou'].mean():.4f} +/- {df['iou'].std():.4f}")
    print(f"  Precision:        {df['precision'].mean():.4f} +/- {df['precision'].std():.4f}")
    print(f"  Recall:           {df['recall'].mean():.4f} +/- {df['recall'].std():.4f}")
    print(f"  Pixel Accuracy:   {df['accuracy'].mean():.4f} +/- {df['accuracy'].std():.4f}")
    print("=" * 55)

    return df


if __name__ == "__main__":
    import argparse
    from src.predict import load_model_from_checkpoint

    parser = argparse.ArgumentParser(description="Evaluate trained U-Net on dataset split")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Dataset split")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, device)

    img_dir = f"data/{args.split}/images"
    msk_dir = f"data/{args.split}/masks"
    out_csv = args.output_csv or f"outputs/{args.split}_metrics.csv"

    evaluate_dataset_and_save_csv(
        model=model,
        images_dir=img_dir,
        masks_dir=msk_dir,
        output_csv=out_csv,
        device=device,
        threshold=args.threshold,
    )

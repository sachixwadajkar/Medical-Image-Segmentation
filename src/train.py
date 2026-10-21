"""Reproducible training pipeline for U-Net histopathology nuclei segmentation."""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import create_dataloaders
from src.evaluate import BCEDiceLoss, evaluate_dataloader
from src.model import UNet


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute one training epoch with gradient backpropagation."""
    model.train()
    running_loss = 0.0
    accumulated_tp = 0
    accumulated_fp = 0
    accumulated_fn = 0
    batch_count = 0

    for images, targets in tqdm(dataloader, desc="Training Batch", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_count += 1

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            p = preds.view(-1).bool()
            t = targets.view(-1).bool()
            accumulated_tp += int(torch.sum(p & t).item())
            accumulated_fp += int(torch.sum(p & (~t)).item())
            accumulated_fn += int(torch.sum((~p) & t).item())

    epoch_loss = running_loss / max(1, batch_count)
    epoch_dice = (2.0 * accumulated_tp) / (2.0 * accumulated_tp + accumulated_fp + accumulated_fn + 1e-7)
    return float(epoch_loss), float(epoch_dice)


def plot_learning_curves(history: Dict[str, List[float]], output_path: str) -> None:
    """Generate high-resolution loss and metric curves over training epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss (BCE + Dice)")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss (BCE + Dice)")
    ax1.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Metrics subplot
    ax2.plot(epochs, history["train_dice"], "b-o", label="Train Dice")
    ax2.plot(epochs, history["val_dice"], "g-^", label="Val Dice")
    ax2.plot(epochs, history["val_iou"], "m--d", label="Val IoU (Jaccard)")
    ax2.set_title("Segmentation Performance (Dice & IoU)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved learning curves to: {output_path}")


def train_pipeline(
    data_dir: str = "data",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    init_features: int = 32,
    bilinear: bool = False,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
    plots_dir: str = "outputs/plots",
    patience: int = 5,
) -> Dict[str, Any]:
    """End-to-end model training, validation, early stopping, and checkpointing."""
    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Detect device and optimize thread count for CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Optimal CPU thread allocation for AMD Ryzen
        torch.set_num_threads(min(6, os.cpu_count() or 4))
    print(f"\n==========================================")
    print(f" U-Net Training Initialized")
    print(f" Compute Device: {device} (threads={torch.get_num_threads()})")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print(f" Base Filters: {init_features} | Bilinear: {bilinear} | Seed: {seed}")
    print(f" Loss Weights: BCE={bce_weight}, Dice={dice_weight}")
    print(f"==========================================\n")

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
    )
    print(f"Dataset Loaded -> Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = UNet(in_channels=3, out_channels=1, init_features=init_features, bilinear=bilinear).to(device)
    print(f"Model Parameters: {model.count_parameters():,}")

    criterion = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
        "lr": [],
    }

    best_val_dice = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train phase
        t_loss, t_dice = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation phase
        val_metrics = evaluate_dataloader(model, val_loader, criterion=criterion, device=device)
        v_loss = val_metrics["loss"]
        v_dice = val_metrics["dice"]
        v_iou = val_metrics["iou"]
        v_prec = val_metrics["precision"]
        v_rec = val_metrics["recall"]

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(v_dice)

        history["train_loss"].append(t_loss)
        history["train_dice"].append(t_dice)
        history["val_loss"].append(v_loss)
        history["val_dice"].append(v_dice)
        history["val_iou"].append(v_iou)
        history["val_precision"].append(v_prec)
        history["val_recall"].append(v_rec)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {t_loss:.4f} | Train Dice: {t_dice:.4f} | "
            f"Val Loss: {v_loss:.4f} | Val Dice: {v_dice:.4f} | Val IoU: {v_iou:.4f} | "
            f"Prec: {v_prec:.4f} | Rec: {v_rec:.4f} | LR: {current_lr:.6f}"
        )

        # Checkpoint if best model
        if v_dice > best_val_dice:
            best_val_dice = v_dice
            best_epoch = epoch
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": v_dice,
                    "val_iou": v_iou,
                    "val_loss": v_loss,
                    "config": {
                        "in_channels": 3,
                        "out_channels": 1,
                        "init_features": init_features,
                        "bilinear": bilinear,
                        "batch_size": batch_size,
                        "lr": lr,
                        "seed": seed,
                    },
                },
                best_ckpt_path,
            )
            print(f"  >>> Best model saved! (Val Dice: {v_dice:.4f} at epoch {epoch})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} consecutive epochs. Stopping at epoch {epoch}.")
                break

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time / 60:.2f} minutes.")
    print(f"Best Validation Dice: {best_val_dice:.4f} achieved at Epoch {best_epoch}.")

    # Save last model checkpoint
    last_ckpt_path = os.path.join(checkpoint_dir, "last_model.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": {
                "in_channels": 3,
                "out_channels": 1,
                "init_features": init_features,
                "bilinear": bilinear,
            },
        },
        last_ckpt_path,
    )

    # Save training history CSV
    history_df = pd.DataFrame(history)
    history_df.index = history_df.index + 1
    history_df.index.name = "epoch"
    history_csv_path = os.path.join(plots_dir, "training_history.csv")
    history_df.to_csv(history_csv_path)
    print(f"Saved history log to: {history_csv_path}")

    # Plot learning curves
    curve_plot_path = os.path.join(plots_dir, "learning_curves.png")
    plot_learning_curves(history, curve_plot_path)

    return {
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "history": history,
        "training_time_seconds": total_training_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on MoNuSeg Nuclei Dataset")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--init_features", type=int, default=16, help="Initial feature channels")
    parser.add_argument("--bilinear", action="store_true", help="Use bilinear upsampling instead of transposed conv")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        init_features=args.init_features,
        bilinear=args.bilinear,
        seed=args.seed,
        patience=args.patience,
    )

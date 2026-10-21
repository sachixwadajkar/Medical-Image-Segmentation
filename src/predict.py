"""Single-image prediction and error visualization module."""

import argparse
import os
import sys
from typing import Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.evaluate import calculate_metrics
from src.model import UNet


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> UNet:
    """Load trained U-Net weights and architecture configuration from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    init_features = config.get("init_features", 16)
    bilinear = config.get("bilinear", True)

    model = UNet(
        in_channels=3,
        out_channels=1,
        init_features=init_features,
        bilinear=bilinear,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def create_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Create a semi-transparent colored mask overlay on top of the original RGB image."""
    overlay = image.copy()
    binary_mask = (mask > 0.5)

    colored_mask = np.zeros_like(image)
    colored_mask[binary_mask] = color

    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    # Highlight boundaries in high-contrast solid line
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 1)
    return overlay


def create_error_map(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> np.ndarray:
    """Create an RGB error map showing True Positives (Green), False Positives (Red), and False Negatives (Blue)."""
    h, w = pred_mask.shape[:2]
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    p = (pred_mask > 0.5)
    g = (gt_mask > 0.5)

    tp = p & g
    fp = p & (~g)
    fn = (~p) & g

    # TP = Green (0, 200, 0)
    error_map[tp] = [0, 200, 0]
    # FP = Red (255, 40, 40)
    error_map[fp] = [255, 40, 40]
    # FN = Blue (40, 100, 255)
    error_map[fn] = [40, 100, 255]

    return error_map


def visualize_prediction(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title_suffix: str = "",
) -> plt.Figure:
    """Generate professional 5-panel visualization:

    1. Original Image
    2. Ground Truth Mask (if provided)
    3. Predicted Mask
    4. Segmentation Overlay
    5. Error Map (TP/FP/FN)
    """
    has_gt = gt_mask is not None
    cols = 5 if has_gt else 3
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]

    # 1. Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original H&E Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    idx = 1
    if has_gt:
        # 2. Ground Truth Mask
        axes[idx].imshow(gt_mask, cmap="gray")
        axes[idx].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
        axes[idx].axis("off")
        idx += 1

    # 3. Predicted Mask
    axes[idx].imshow(pred_mask, cmap="gray")
    axes[idx].set_title("Predicted Mask", fontsize=11, fontweight="bold")
    axes[idx].axis("off")
    idx += 1

    # 4. Overlay
    overlay = create_segmentation_overlay(image, pred_mask, color=(0, 255, 0), alpha=0.4)
    axes[idx].imshow(overlay)
    axes[idx].set_title("Segmentation Overlay", fontsize=11, fontweight="bold")
    axes[idx].axis("off")
    idx += 1

    # 5. Error Map
    if has_gt:
        error_map = create_error_map(pred_mask, gt_mask)
        axes[idx].imshow(error_map)
        metrics = calculate_metrics(pred_mask, gt_mask)
        axes[idx].set_title(
            f"Error Analysis\nDice: {metrics['dice']:.3f} | IoU: {metrics['iou']:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        axes[idx].axis("off")

    plt.suptitle(f"Nuclei Segmentation Analysis {title_suffix}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")

    return fig


def predict_single_image(
    image_path: str,
    checkpoint_path: str = "checkpoints/best_model.pth",
    gt_path: Optional[str] = None,
    output_path: Optional[str] = None,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute prediction on a single image and save the 5-panel diagnostic figure."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(checkpoint_path, device)

    pil_img = Image.open(image_path).convert("RGB")
    img_arr = np.array(pil_img)
    norm_img = img_arr.astype(np.float32) / 255.0

    tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        prob_map = model.predict_probability(tensor_img).squeeze().cpu().numpy()
        pred_mask = (prob_map > threshold).astype(np.float32)

    gt_mask = None
    if gt_path and os.path.exists(gt_path):
        gt_mask = (np.array(Image.open(gt_path).convert("L"), dtype=np.float32) > 127.5).astype(np.float32)

    visualize_prediction(
        image=img_arr,
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        output_path=output_path,
        title_suffix=f"({os.path.basename(image_path)})",
    )
    plt.close()

    return prob_map, pred_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict nuclei segmentation on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to ground truth mask")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Checkpoint file")
    parser.add_argument("--output", type=str, default="outputs/predictions/sample_prediction.png", help="Output figure")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    args = parser.parse_args()

    predict_single_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        gt_path=args.mask,
        output_path=args.output,
        threshold=args.threshold,
    )

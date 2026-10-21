"""High-Resolution Tiled Inference and Cellular Quantification Pipeline.

Supports sliding-window patch inference with 2D Hann window probability blending,
eliminating tile border artifacts on digital pathology images. Also provides connected
component nuclei quantification with morphological metrics.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image

from src.evaluate import calculate_metrics
from src.predict import (
    load_model_from_checkpoint,
    create_segmentation_overlay,
    create_error_map,
)


def create_2d_window(tile_size: int, window_type: str = "hann") -> np.ndarray:
    """Generate a 2D smoothing window to blend tile borders seamlessly.

    Tapers probabilities towards tile edges to suppress edge discontinuities.
    """
    if window_type == "hann":
        win1d = np.hanning(tile_size)
        win2d = np.outer(win1d, win1d)
        # Avoid zero weights at the exact boundary
        win2d = np.clip(win2d, 1e-4, 1.0)
    elif window_type == "linear":
        ramp = np.linspace(0.1, 1.0, tile_size // 2)
        win1d = np.concatenate([ramp, ramp[::-1]])
        win2d = np.outer(win1d, win1d)
    else:
        win2d = np.ones((tile_size, tile_size), dtype=np.float32)
    return win2d.astype(np.float32)


def get_sliding_window_coords(
    img_h: int,
    img_w: int,
    tile_size: int = 256,
    overlap: int = 64,
) -> List[Tuple[int, int]]:
    """Compute top-left coordinates for sliding window tiling with clamped borders."""
    stride = tile_size - overlap
    assert stride > 0, "Overlap must be strictly less than tile_size"

    y_coords = list(range(0, max(1, img_h - tile_size + 1), stride))
    if not y_coords or y_coords[-1] + tile_size < img_h:
        y_coords.append(img_h - tile_size)

    x_coords = list(range(0, max(1, img_w - tile_size + 1), stride))
    if not x_coords or x_coords[-1] + tile_size < img_w:
        x_coords.append(img_w - tile_size)

    # Clean duplicates and clamp negatives for small images
    y_coords = sorted(list(set([max(0, y) for y in y_coords])))
    x_coords = sorted(list(set([max(0, x) for x in x_coords])))

    coords = [(y, x) for y in y_coords for x in x_coords]
    return coords


def tiled_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int = 256,
    overlap: int = 64,
    batch_size: int = 4,
    blend: bool = True,
    threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Execute high-resolution sliding-window inference with probability blending.

    Args:
        model: Trained PyTorch U-Net model.
        image: Full resolution RGB image of shape (H, W, 3).
        tile_size: Patch dimension (default: 256).
        overlap: Border overlap in pixels (default: 64).
        batch_size: Number of tiles processed in parallel.
        blend: If True, uses 2D Hann window blending to remove boundary artifacts.
        threshold: Decision threshold for binary segmentation.
        device: Torch compute device.

    Returns:
        Dictionary with probability map, binary mask, tile count, runtime, and memory metrics.
    """
    model.eval()
    h, w, c = image.shape
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()

    # Pre-pad if image is smaller than tile_size
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h > 0 or pad_w > 0:
        working_img = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        working_img = image

    work_h, work_w = working_img.shape[:2]
    coords = get_sliding_window_coords(work_h, work_w, tile_size, overlap)
    total_tiles = len(coords)

    # Weighting window
    window = create_2d_window(tile_size, "hann" if blend else "uniform")

    # Accumulators
    prob_accum = np.zeros((work_h, work_w), dtype=np.float32)
    weight_accum = np.zeros((work_h, work_w), dtype=np.float32)

    # Process tiles in mini-batches
    for i in range(0, total_tiles, batch_size):
        batch_coords = coords[i : i + batch_size]
        tile_tensors = []

        for y, x in batch_coords:
            patch = working_img[y : y + tile_size, x : x + tile_size].astype(np.float32) / 255.0
            # (H, W, 3) -> (3, H, W)
            tensor = torch.from_numpy(patch).permute(2, 0, 1)
            tile_tensors.append(tensor)

        batch_tensor = torch.stack(tile_tensors, dim=0).to(device)

        with torch.no_grad():
            preds = model.predict_probability(batch_tensor).squeeze(1).cpu().numpy()

        for j, (y, x) in enumerate(batch_coords):
            tile_pred = preds[j]
            prob_accum[y : y + tile_size, x : x + tile_size] += tile_pred * window
            weight_accum[y : y + tile_size, x : x + tile_size] += window

    # Normalize accumulated probabilities by accumulated weights
    weight_accum = np.maximum(weight_accum, 1e-5)
    blended_prob = prob_accum / weight_accum

    # Crop out padding
    blended_prob = blended_prob[:h, :w]
    binary_mask = (blended_prob > threshold).astype(np.float32)

    dt = time.perf_counter() - t0
    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_delta = mem_after - mem_before

    return {
        "probability_map": blended_prob,
        "binary_mask": binary_mask,
        "tile_count": total_tiles,
        "tile_size": tile_size,
        "overlap": overlap,
        "inference_time_seconds": dt,
        "peak_ram_mb": mem_after,
        "ram_delta_mb": max(0.0, mem_delta),
        "resolution": (h, w),
    }


def quantify_nuclei(
    binary_mask: np.ndarray,
    microns_per_pixel: float = 0.25,
    min_area_pixels: int = 15,
    max_area_pixels: int = 3500,
) -> Dict[str, Any]:
    """Perform connected components nuclei quantification and morphological analysis.

    Distinguishes binary semantic segmentation from instance segmentation:
    Adjoining/touching nuclei in close contact are grouped into single connected components
    unless separated by instance post-processing (e.g. Watershed / HoVer-Net).

    Args:
        binary_mask: 2D binary array {0, 1}
        microns_per_pixel: Pixel spatial calibration (MoNuSeg TCGA 40x is ~0.25 um/px)
        min_area_pixels: Minimum area threshold to reject noise
        max_area_pixels: Maximum area threshold to reject large tissue fold artifacts

    Returns:
        Dictionary with count, areas, summary stats, labeled mask, and DataFrame.
    """
    uint_mask = (binary_mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        uint_mask, connectivity=8
    )

    records = []
    valid_component_idx = 1
    clean_labeled_mask = np.zeros_like(labels, dtype=np.int32)

    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        if min_area_pixels <= area_px <= max_area_pixels:
            cx, cy = centroids[i]
            area_um2 = area_px * (microns_per_pixel ** 2)
            records.append({
                "component_id": valid_component_idx,
                "centroid_x": round(float(cx), 1),
                "centroid_y": round(float(cy), 1),
                "area_pixels": int(area_px),
                "area_microns2": round(float(area_um2), 2),
                "bounding_width": int(stats[i, cv2.CC_STAT_WIDTH]),
                "bounding_height": int(stats[i, cv2.CC_STAT_HEIGHT]),
            })
            clean_labeled_mask[labels == i] = valid_component_idx
            valid_component_idx += 1

    df = pd.DataFrame(records)
    total_nuclei = len(records)

    summary_stats = {
        "total_detected_nuclei": total_nuclei,
        "microns_per_pixel": microns_per_pixel,
        "mean_area_pixels": round(float(df["area_pixels"].mean()), 1) if total_nuclei > 0 else 0,
        "median_area_pixels": round(float(df["area_pixels"].median()), 1) if total_nuclei > 0 else 0,
        "std_area_pixels": round(float(df["area_pixels"].std()), 1) if total_nuclei > 1 else 0,
        "mean_area_um2": round(float(df["area_microns2"].mean()), 2) if total_nuclei > 0 else 0,
        "total_nuclei_area_um2": round(float(df["area_microns2"].sum()), 2) if total_nuclei > 0 else 0,
    }

    return {
        "summary": summary_stats,
        "components_df": df,
        "labeled_mask": clean_labeled_mask,
    }


def visualize_high_res_results(
    image: np.ndarray,
    pred_mask: np.ndarray,
    quant_result: Dict[str, Any],
    gt_mask: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    benchmarks: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """Generate a multi-panel digital pathology diagnostic figure."""
    has_gt = gt_mask is not None
    cols = 4 if has_gt else 3
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

    # 1. Original
    axes[0].imshow(image)
    bench_str = f"\nTime: {benchmarks['inference_time_seconds']:.2f}s | Tiles: {benchmarks['tile_count']}" if benchmarks else ""
    axes[0].set_title(f"High-Res Histopathology{bench_str}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Predicted Mask & Overlay
    overlay = create_segmentation_overlay(image, pred_mask, color=(0, 255, 0), alpha=0.45)
    axes[1].imshow(overlay)
    axes[1].set_title("U-Net Tiled Segmentation Overlay", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Labeled Instance Components
    labeled = quant_result["labeled_mask"]
    # Pseudocolor labeled mask
    hsv = np.zeros((*labeled.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (labeled * 37) % 180  # Vary hue
    hsv[..., 1] = np.where(labeled > 0, 255, 0)
    hsv[..., 2] = np.where(labeled > 0, 255, 0)
    rgb_labeled = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    axes[2].imshow(rgb_labeled)
    q_sum = quant_result["summary"]
    axes[2].set_title(
        f"Quantified Nuclei ({q_sum['total_detected_nuclei']} detected)\nMean Area: {q_sum['mean_area_um2']} \u03bcm\u00b2",
        fontsize=11,
        fontweight="bold",
    )
    axes[2].axis("off")

    # 4. Error Map (if GT available)
    if has_gt:
        error_map = create_error_map(pred_mask, gt_mask)
        metrics = calculate_metrics(pred_mask, gt_mask)
        axes[3].imshow(error_map)
        axes[3].set_title(
            f"Whole-Slide Error Map\nDice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}",
            fontsize=11,
            fontweight="bold",
        )
        axes[3].axis("off")

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved high-resolution visualization to: {output_path}")

    return fig


def run_high_res_inference(
    image_path: str,
    checkpoint_path: str = "checkpoints/best_model.pth",
    gt_path: Optional[str] = None,
    output_dir: str = "outputs/predictions",
    tile_size: int = 256,
    overlap: int = 64,
    threshold: float = 0.5,
    microns_per_pixel: float = 0.25,
) -> Dict[str, Any]:
    """Execute end-to-end tiled inference, quantification, and reporting on a high-res image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(checkpoint_path, device)

    pil_img = Image.open(image_path).convert("RGB")
    img_arr = np.array(pil_img)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    gt_mask = None
    if gt_path and os.path.exists(gt_path):
        gt_mask = (np.array(Image.open(gt_path).convert("L"), dtype=np.float32) > 127.5).astype(np.float32)

    # 1. Tiled Inference
    benchmarks = tiled_inference(
        model=model,
        image=img_arr,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=4,
        blend=True,
        threshold=threshold,
        device=device,
    )
    pred_mask = benchmarks["binary_mask"]

    # 2. Quantification
    quant_result = quantify_nuclei(
        binary_mask=pred_mask,
        microns_per_pixel=microns_per_pixel,
    )

    # 3. Export CSV and Figure
    fig_path = os.path.join(output_dir, f"{base_name}_tiled_inference.png")
    csv_path = os.path.join(output_dir, f"{base_name}_nuclei_quantification.csv")

    quant_result["components_df"].to_csv(csv_path, index=False)
    visualize_high_res_results(
        image=img_arr,
        pred_mask=pred_mask,
        quant_result=quant_result,
        gt_mask=gt_mask,
        output_path=fig_path,
        benchmarks=benchmarks,
    )
    plt.close()

    print(f"\nHigh-Res Inference Benchmark on {base_name}:")
    print(f"  Resolution: {benchmarks['resolution'][0]} x {benchmarks['resolution'][1]}")
    print(f"  Tile Size: {tile_size} x {tile_size} (Overlap: {overlap}px)")
    print(f"  Total Tiles Processed: {benchmarks['tile_count']}")
    print(f"  Runtime: {benchmarks['inference_time_seconds']:.2f} seconds")
    print(f"  Detected Nuclei: {quant_result['summary']['total_detected_nuclei']}")
    print(f"  Quantification CSV: {csv_path}")

    return {
        "benchmarks": benchmarks,
        "quantification": quant_result["summary"],
        "figure_path": fig_path,
        "csv_path": csv_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-resolution tiled inference with quantification")
    parser.add_argument("--image", type=str, required=True, help="Path to high-res image")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to ground-truth mask")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Checkpoint file")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions", help="Output directory")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size")
    parser.add_argument("--overlap", type=int, default=64, help="Tile overlap in pixels")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    run_high_res_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        gt_path=args.mask,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=args.threshold,
    )

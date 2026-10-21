"""Data preparation and preprocessing pipeline for MoNuSeg histopathology dataset.

Converts XML polygon annotations to binary ground-truth masks, performs patient-level
data splitting to prevent data leakage, extracts normalized image patches, and validates
image-mask correspondence.
"""

import os
import glob
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def parse_monuseg_xml(xml_path: str, target_shape: Tuple[int, int] = (1000, 1000)) -> Tuple[np.ndarray, int]:
    """Parse MoNuSeg XML polygon annotations and render a binary mask.

    Args:
        xml_path: Path to the annotation XML file.
        target_shape: Expected (height, width) of the binary mask.

    Returns:
        mask: uint8 binary mask with values 0 (background) and 255 (nuclei).
        num_valid_regions: Total number of valid polygonal nuclei rendered.

    Raises:
        FileNotFoundError: If the XML file does not exist.
        ET.ParseError: If the XML file is malformed.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Annotation file not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    mask = np.zeros(target_shape, dtype=np.uint8)
    regions = root.findall(".//Region")
    num_valid_regions = 0

    for region in regions:
        vertices = region.findall(".//Vertex")
        coords = []
        for vertex in vertices:
            try:
                x = float(vertex.get("X", 0.0))
                y = float(vertex.get("Y", 0.0))
                coords.append([x, y])
            except (ValueError, TypeError):
                continue

        # A valid closed polygon must have at least 3 vertices
        if len(coords) >= 3:
            pts = np.array(coords, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            num_valid_regions += 1

    return mask, num_valid_regions


def extract_patient_id(filename: str) -> str:
    """Extract patient identifier from TCGA barcode filename.

    Example: 'TCGA-18-5592-01Z-00-DX1.tif' -> 'TCGA-18-5592'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("-")
    if len(parts) >= 3:
        return f"{parts[0]}-{parts[1]}-{parts[2]}"
    return base


def get_image_mask_pairs(raw_dir: str) -> List[Tuple[str, str, str]]:
    """Scan raw directory and pair each tissue image with its corresponding XML annotation.

    Returns:
        List of tuples: (image_path, xml_path, patient_id)
    """
    # MoNuSeg training data typically has separate Annotations and Tissue Images folders
    # or all files together in one folder (as in test data).
    tif_files = glob.glob(os.path.join(raw_dir, "**", "*.tif"), recursive=True)
    xml_files = glob.glob(os.path.join(raw_dir, "**", "*.xml"), recursive=True)

    xml_map = {os.path.splitext(os.path.basename(f))[0]: f for f in xml_files}
    pairs = []

    for tif in tif_files:
        base = os.path.splitext(os.path.basename(tif))[0]
        if base in xml_map:
            patient_id = extract_patient_id(base)
            pairs.append((tif, xml_map[base], patient_id))

    return sorted(pairs, key=lambda x: x[0])


def extract_patches_from_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 248,
) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:
    """Extract overlapping or tiled patches that completely cover the canvas.

    Guarantees full coverage of boundary pixels by clamping the final step.

    Args:
        image: Source RGB image array of shape (H, W, 3).
        mask: Source binary mask array of shape (H, W).
        patch_size: Square patch dimension in pixels.
        stride: Step size between patches.

    Returns:
        List of (image_patch, mask_patch, (y, x, patch_h, patch_w))
    """
    h, w = image.shape[:2]
    y_coords = list(range(0, h - patch_size + 1, stride))
    if not y_coords or y_coords[-1] + patch_size < h:
        y_coords.append(h - patch_size)

    x_coords = list(range(0, w - patch_size + 1, stride))
    if not x_coords or x_coords[-1] + patch_size < w:
        x_coords.append(w - patch_size)

    y_coords = sorted(list(set(y_coords)))
    x_coords = sorted(list(set(x_coords)))

    patches = []
    for y in y_coords:
        for x in x_coords:
            img_p = image[y : y + patch_size, x : x + patch_size]
            msk_p = mask[y : y + patch_size, x : x + patch_size]
            patches.append((img_p, msk_p, (y, x, patch_size, patch_size)))

    return patches


def prepare_monuseg_data(
    raw_dir: str = "data/raw",
    output_dir: str = "data",
    patch_size: int = 256,
    stride: int = 248,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end dataset preprocessing with patient-level splitting.

    1. Discovers image-XML pairs.
    2. Groups and splits at the patient level to prevent data leakage.
    3. Converts XML annotations to binary masks.
    4. Extracts patches and saves them as PNG files.
    5. Saves full ground truth masks alongside raw images for whole-slide evaluation.
    6. Produces a detailed statistical summary.
    """
    train_raw_folder = os.path.join(raw_dir, "MoNuSeg 2018 Training Data")
    if not os.path.exists(train_raw_folder):
        train_raw_folder = raw_dir

    pairs = get_image_mask_pairs(train_raw_folder)
    if not pairs:
        raise RuntimeError(f"No valid image-XML pairs found in {train_raw_folder}")

    # Group by patient ID
    patient_to_pairs: Dict[str, List[Tuple[str, str, str]]] = {}
    for item in pairs:
        patient_to_pairs.setdefault(item[2], []).append(item)

    unique_patients = sorted(list(patient_to_pairs.keys()))
    train_pts, val_pts = train_test_split(
        unique_patients, test_size=val_ratio, random_state=seed, shuffle=True
    )

    splits = {
        "train": [p for pt in train_pts for p in patient_to_pairs[pt]],
        "val": [p for pt in val_pts for p in patient_to_pairs[pt]],
    }

    # Also handle independent MoNuSeg test set if present
    test_raw_folder = os.path.join(raw_dir, "MoNuSegTestData")
    if os.path.exists(test_raw_folder):
        test_pairs = get_image_mask_pairs(test_raw_folder)
        if test_pairs:
            splits["test"] = test_pairs

    summary: Dict[str, Any] = {
        "patch_size": patch_size,
        "stride": stride,
        "random_seed": seed,
        "val_ratio": val_ratio,
        "splits": {},
    }

    for split_name, split_pairs in splits.items():
        img_out_dir = os.path.join(output_dir, split_name, "images")
        msk_out_dir = os.path.join(output_dir, split_name, "masks")
        full_msk_dir = os.path.join(output_dir, split_name, "full_masks")
        full_img_dir = os.path.join(output_dir, split_name, "full_images")

        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(msk_out_dir, exist_ok=True)
        os.makedirs(full_msk_dir, exist_ok=True)
        os.makedirs(full_img_dir, exist_ok=True)

        total_source_images = len(split_pairs)
        total_nuclei = 0
        total_patches = 0
        empty_patches = 0
        total_pixels = 0
        foreground_pixels = 0

        print(f"\n--- Processing '{split_name}' split ({total_source_images} source images) ---")

        for img_path, xml_path, pt_id in split_pairs:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            pil_img = Image.open(img_path).convert("RGB")
            img_arr = np.array(pil_img)

            mask_arr, n_nuclei = parse_monuseg_xml(xml_path, target_shape=img_arr.shape[:2])
            total_nuclei += n_nuclei

            # Save full image and full ground truth mask for high-resolution evaluation
            Image.fromarray(img_arr).save(os.path.join(full_img_dir, f"{base_name}.png"))
            Image.fromarray(mask_arr).save(os.path.join(full_msk_dir, f"{base_name}.png"))

            # Extract patches
            patches = extract_patches_from_canvas(img_arr, mask_arr, patch_size=patch_size, stride=stride)
            for idx, (img_p, msk_p, bbox) in enumerate(patches):
                patch_name = f"{base_name}_patch_{idx:03d}_{bbox[0]}_{bbox[1]}.png"
                img_p_path = os.path.join(img_out_dir, patch_name)
                msk_p_path = os.path.join(msk_out_dir, patch_name)

                Image.fromarray(img_p).save(img_p_path)
                Image.fromarray(msk_p).save(msk_p_path)

                total_patches += 1
                fg = int(np.count_nonzero(msk_p))
                if fg == 0:
                    empty_patches += 1
                foreground_pixels += fg
                total_pixels += msk_p.size

        fg_percentage = (foreground_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

        split_stat = {
            "source_images": total_source_images,
            "patient_count": len(set(p[2] for p in split_pairs)),
            "total_annotated_nuclei": total_nuclei,
            "total_patches_extracted": total_patches,
            "empty_mask_patches": empty_patches,
            "foreground_nuclei_pixel_ratio": round(fg_percentage, 2),
        }
        summary["splits"][split_name] = split_stat

        print(f"[{split_name.upper()}] Images: {total_source_images} | Nuclei: {total_nuclei} | "
              f"Patches: {total_patches} (Empty: {empty_patches}) | Nuclei Coverage: {fg_percentage:.2f}%")

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPreprocessing complete! Summary metadata written to: {summary_path}")
    return summary


if __name__ == "__main__":
    prepare_monuseg_data(raw_dir="data/raw", output_dir="data", patch_size=256, stride=248, val_ratio=0.2, seed=42)

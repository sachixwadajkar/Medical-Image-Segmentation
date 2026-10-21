"""Medical Image Segmentation Web Application (Streamlit).

Interactive digital pathology UI for cell nuclei segmentation, high-resolution tiled
inference, probability blending, and morphological nuclei quantification.
"""

import io
import os
import sys
import time
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.inference import tiled_inference, quantify_nuclei
from src.model import UNet
from src.predict import (
    create_segmentation_overlay,
    create_error_map,
    load_model_from_checkpoint,
)


# Configure Streamlit page
st.set_page_config(
    page_title="MoNuSeg U-Net | Digital Pathology Nuclei Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Medical/Scientific Application
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading U-Net Checkpoint...")
def get_cached_model(checkpoint_path: str):
    """Load model once into memory and cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        return None, None
    model = load_model_from_checkpoint(checkpoint_path, device)
    return model, device


def main():
    st.markdown('<div class="main-header">🔬 Digital Pathology Nuclei Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">PyTorch U-Net for Automated Nuclear Boundary Delineation in Multi-Organ H&E Histopathology</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Educational & Research Prototype Notice:</strong> This software is engineered strictly
            for research and portfolio demonstration (aligned with AIRA Matrix digital pathology competencies).
            It is not certified or validated for clinical patient diagnosis or treatment planning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Configuration
    st.sidebar.header("⚙️ Model Configuration")
    ckpt_path = st.sidebar.text_input("Checkpoint Path", value="checkpoints/best_model.pth")
    threshold = st.sidebar.slider("Probability Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    st.sidebar.markdown("---")
    st.sidebar.header("🧩 High-Res Tiled Inference")
    use_tiling = st.sidebar.checkbox("Enable Sliding-Window Tiling", value=True, help="Recommended for images larger than 256x256 to avoid resizing distortion.")
    tile_size = st.sidebar.select_slider("Tile Dimension (px)", options=[128, 256, 512], value=256)
    overlap = st.sidebar.slider("Tile Overlap (px)", min_value=16, max_value=128, value=64, step=16)
    use_blending = st.sidebar.checkbox("2D Hann Window Blending", value=True, help="Smooths probability seams between adjacent overlapping tiles.")

    st.sidebar.markdown("---")
    st.sidebar.header("📏 Spatial Calibration")
    mpp = st.sidebar.number_input("Microns per Pixel (\u03bcm/px)", value=0.25, step=0.01, format="%.3f")

    # Load Model
    model, device = get_cached_model(ckpt_path)

    if model is None:
        st.warning(f"⚠️ Model checkpoint `{ckpt_path}` was not found. Please train the model first using `python src/train.py`.")
        st.info("You can still explore pre-generated test sample images below once training finishes.")
        return

    st.sidebar.success(f"Model active on: **{device.type.upper()}**")

    # Image Input Section
    st.subheader("1. Histopathology Image Input")
    col_input, col_preset = st.columns([2, 1])

    with col_preset:
        preset_options = ["None (Upload my own)"]
        sample_files = []
        if os.path.exists("data/val/images"):
            sample_files = sorted(os.listdir("data/val/images"))[:5]
        if os.path.exists("data/test/full_images"):
            sample_files += sorted(os.listdir("data/test/full_images"))[:3]

        preset_options.extend(sample_files)
        selected_preset = st.selectbox("Or choose a validation sample:", options=preset_options)

    with col_input:
        uploaded_file = st.file_uploader(
            "Upload an H&E histopathology image (TIFF, PNG, JPG):",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
        )

    # Determine Active Image
    pil_image = None
    image_name = "Uploaded_Image"

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_name = uploaded_file.name
    elif selected_preset != "None (Upload my own)":
        image_name = selected_preset
        preset_path = None
        for candidate_dir in ["data/val/images", "data/test/full_images", "data/train/images"]:
            p = os.path.join(candidate_dir, selected_preset)
            if os.path.exists(p):
                preset_path = p
                break
        if preset_path:
            pil_image = Image.open(preset_path).convert("RGB")

    if pil_image is None:
        st.info("👆 Please upload an H&E tissue image or select a validation sample to initiate segmentation.")
        return

    img_arr = np.array(pil_image)
    orig_h, orig_w = img_arr.shape[:2]

    # Processing and Inference
    with st.spinner("Executing neural network inference..."):
        t_start = time.perf_counter()

        if use_tiling and (orig_h > 256 or orig_w > 256):
            benchmarks = tiled_inference(
                model=model,
                image=img_arr,
                tile_size=tile_size,
                overlap=overlap,
                batch_size=4,
                blend=use_blending,
                threshold=threshold,
                device=device,
            )
            prob_map = benchmarks["probability_map"]
            binary_mask = benchmarks["binary_mask"]
            tile_count = benchmarks["tile_count"]
        else:
            norm_img = img_arr.astype(np.float32) / 255.0
            t_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                prob_map = model.predict_probability(t_img).squeeze().cpu().numpy()
            binary_mask = (prob_map > threshold).astype(np.float32)
            tile_count = 1

        inference_time = time.perf_counter() - t_start

    # Cellular Quantification
    quant = quantify_nuclei(binary_mask, microns_per_pixel=mpp)
    q_sum = quant["summary"]

    # Metrics Summary Row
    st.subheader("2. Segmentation & Quantification Results")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Image Dimensions", f"{orig_w} \u00d7 {orig_h} px")
    with m_col2:
        st.metric("Inference Runtime", f"{inference_time:.2f} s", delta=f"{tile_count} tiles" if use_tiling else "Single pass")
    with m_col3:
        st.metric("Nuclei Detected", f"{q_sum['total_detected_nuclei']}")
    with m_col4:
        st.metric("Mean Nuclear Area", f"{q_sum['mean_area_um2']} \u03bcm\u00b2")

    # Visual Inspection Panels
    tab1, tab2, tab3 = st.tabs(["🖼️ Visual Segmentation Panels", "📊 Nuclear Morphometry Data", "🔬 Ground Truth Comparison"])

    overlay = create_segmentation_overlay(img_arr, binary_mask, color=(0, 255, 0), alpha=0.45)

    with tab1:
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            st.image(img_arr, caption=f"Original H&E ({orig_w}x{orig_h})", use_container_width=True)
        with v_col2:
            st.image(binary_mask, caption=f"Predicted Binary Mask (Threshold: {threshold})", use_container_width=True)
        with v_col3:
            st.image(overlay, caption="Segmentation Overlay (Nuclei Highlighted in Green)", use_container_width=True)

    with tab2:
        st.markdown("#### Individual Detected Nuclei Measurements")
        df = quant["components_df"]
        if not df.empty:
            st.dataframe(df, use_container_width=True, height=280)

            # Nuclear size histogram
            fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
            ax_hist.hist(df["area_microns2"], bins=25, color="#3B82F6", edgecolor="black", alpha=0.7)
            ax_hist.set_title("Nuclear Area Distribution (\u03bcm\u00b2)", fontsize=11, fontweight="bold")
            ax_hist.set_xlabel("Area (\u03bcm\u00b2)")
            ax_hist.set_ylabel("Nuclei Count")
            ax_hist.grid(True, linestyle=":", alpha=0.5)
            st.pyplot(fig_hist)
            plt.close()

            # CSV Download button
            csv_buf = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Nuclei Morphometry CSV",
                data=csv_buf,
                file_name=f"{os.path.splitext(image_name)[0]}_nuclei_stats.csv",
                mime="text/csv",
            )
        else:
            st.info("No nuclei detected above the minimum area threshold.")

    with tab3:
        # Check if matching ground truth exists
        gt_mask = None
        base_stem = os.path.splitext(image_name)[0]
        for gt_dir in ["data/val/masks", "data/test/full_masks", "data/val/full_masks"]:
            candidate = os.path.join(gt_dir, f"{base_stem}.png")
            if os.path.exists(candidate):
                gt_mask = (np.array(Image.open(candidate).convert("L")) > 127.5).astype(np.float32)
                break

        if gt_mask is not None:
            from src.evaluate import calculate_metrics
            metrics = calculate_metrics(binary_mask, gt_mask)
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.image(gt_mask, caption="Official Ground Truth Annotation Mask", use_container_width=True)
            with g_col2:
                err_map = create_error_map(binary_mask, gt_mask)
                st.image(err_map, caption="Error Diagnostic (Green: TP, Red: FP, Blue: FN)", use_container_width=True)

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Dice Score", f"{metrics['dice']:.4f}")
            e2.metric("IoU (Jaccard)", f"{metrics['iou']:.4f}")
            e3.metric("Precision", f"{metrics['precision']:.4f}")
            e4.metric("Recall", f"{metrics['recall']:.4f}")
        else:
            st.info("No ground-truth XML mask found for this image to compute quantitative error metrics.")

    # Image Downloads
    st.markdown("---")
    st.subheader("3. Export Model Predictions")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
        buf_mask = io.BytesIO()
        mask_pil.save(buf_mask, format="PNG")
        st.download_button(
            label="💾 Download Binary Mask (PNG)",
            data=buf_mask.getvalue(),
            file_name=f"{os.path.splitext(image_name)[0]}_mask.png",
            mime="image/png",
        )

    with d_col2:
        overlay_pil = Image.fromarray(overlay)
        buf_overlay = io.BytesIO()
        overlay_pil.save(buf_overlay, format="PNG")
        st.download_button(
            label="💾 Download Segmentation Overlay (PNG)",
            data=buf_overlay.getvalue(),
            file_name=f"{os.path.splitext(image_name)[0]}_overlay.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()

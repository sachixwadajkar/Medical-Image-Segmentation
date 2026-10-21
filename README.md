# Medical Image Segmentation Using Deep Learning (PyTorch U-Net)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](app.py)
[![Unit Tests](https://img.shields.io/badge/pytest-20%20passed-brightgreen.svg)](tests/)

An end-to-end, production-grade deep learning pipeline for automated cell nuclei segmentation in hematoxylin and eosin (H&E) stained histopathology images using a custom PyTorch U-Net architecture. Developed with digital pathology engineering rigor, zero data leakage partitioning, high-resolution sliding-window inference with 2D Hann window blending, morphological cellular quantification, and an interactive Streamlit application.

---

## 1. Clinical Motivation & Engineering Objectives

In computational pathology and precision oncology, accurate segmentation of cell nuclei is the foundational prerequisite for downstream quantitative biomarker discovery, including:
- **Tumor cellularity and mitotic rate estimation** for tumor grading.
- **Nuclear morphometry** (nuclear pleomorphism, area distribution, eccentricity, chromatin texture).
- **Spatial tumor-infiltrating lymphocyte (TIL) quantification** for immunotherapy response prediction.

Manual annotation by pathologists is laborious, subjective, and subject to inter-observer variability. This project provides a robust, reproducible automated segmentation pipeline aligned with the core competencies of **AIRA Matrix Trainee Engineer – Artificial Intelligence**:
1. **Digital Pathology & WSI Processing**: Handling high-resolution tissue sections via patch extraction and seamless sliding-window reconstruction.
2. **Medical Computer Vision**: Multi-organ nuclear boundary delineation under severe stain variations and high visual clutter.
3. **Mathematical Loss Engineering**: Alleviating extreme foreground/background class imbalance through combined Binary Cross-Entropy and soft Dice loss.
4. **Clean Code & Software Testing**: Modular architecture with 20 unit tests covering data loaders, tensor transformations, model gradients, metrics, and sliding-window coverage.

---

## 2. Dataset: MoNuSeg Challenge

The pipeline utilizes the official **Multi-Organ Nucleus Segmentation (MoNuSeg)** benchmark:
- **Images**: Sourced from The Cancer Genome Atlas (TCGA) spanning diverse organ sites (breast, kidney, liver, prostate, bladder, colon, stomach).
- **Scale**: Captured at $40\times$ optical magnification (~$0.25\ \mu\text{m/pixel}$).
- **Annotations**: Over 30,000 manually traced nuclear boundary polygons by expert pathologists stored as Aperio ImageScope XML files.

### Zero-Data-Leakage Patient-Level Split
In digital pathology, slide fixation and staining create patient-specific visual artifacts. Splitting patches randomly after extraction causes severe **data leakage** because patches from the same patient appear in both train and validation sets. 

Here, **data splitting is performed strictly at the patient/source-image level before patch extraction**:

| Dataset Split | Source Images (Patients) | Annotated Nuclei | $256 \times 256$ Patches | Empty Patches | Nuclei Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 29 | 18,649 | 464 | 0 | **27.12%** |
| **Validation** | 8 | 5,486 | 128 | 0 | **25.40%** |
| **Independent Test** | 14 | 6,697 | 224 | 0 | **21.69%** |
| **Total** | **51** | **30,832** | **816** | **0** | **25.34%** |

---

## 3. U-Net Model Architecture

The U-Net architecture is implemented from scratch in PyTorch (`src/model.py`), featuring a contracting path (encoder), bottleneck, and expanding path (decoder) with lateral skip connections:

```
Input RGB Patch (3, 256, 256)
  │
  ├──> [DoubleConv 3 -> 16] ─────────────────── (Skip 1) ───────────────────┐
  │         │ MaxPool2d (2x2)                                                │
  │         ▼                                                                │
  ├──> [DoubleConv 16 -> 32] ────────────────── (Skip 2) ─────────────┐      │
  │         │ MaxPool2d (2x2)                                         │      │
  │         ▼                                                         │      │
  ├──> [DoubleConv 32 -> 64] ────────────────── (Skip 3) ──────┐      │      │
  │         │ MaxPool2d (2x2)                                  │      │      │
  │         ▼                                                  │      │      │
  ├──> [DoubleConv 64 -> 128] ───────────────── (Skip 4) ┐     │      │      │
  │         │ MaxPool2d (2x2)                            │     │      │      │
  │         ▼                                            │     │      │      │
  └──> [Bottleneck DoubleConv 128 -> 128]                │     │      │      │
            │                                            │     │      │      │
            ▼ Upsample (2x)                              │     │      │      │
       [Cat + DoubleConv 256 -> 64] <────────────────────┘     │      │      │
            │ Upsample (2x)                                    │      │      │
            ▼                                                  │      │      │
       [Cat + DoubleConv 128 -> 32] <──────────────────────────┘      │      │
            │ Upsample (2x)                                           │      │
            ▼                                                         │      │
       [Cat + DoubleConv 64 -> 16]  <─────────────────────────────────┘      │
            │ Upsample (2x)                                                  │
            ▼                                                                │
       [Cat + DoubleConv 32 -> 16]  <────────────────────────────────────────┘
            │
            ▼ Conv2d (1x1)
       Raw Output Logits (1, 256, 256)
```

- **Trainable Parameters**: 838,785 parameters.
- **Double Convolution Blocks**: Each block consists of `Conv2d(3x3, pad=1)` $\to$ `BatchNorm2d` $\to$ `ReLU(inplace=True)` $\to$ `Conv2d(3x3, pad=1)` $\to$ `BatchNorm2d` $\to$ `ReLU(inplace=True)`.
- **Skip Connections**: Concatenate high-resolution spatial feature maps from the encoder to the decoder, recovering sharp boundary localization lost during pooling.
- **Output**: Returns unnormalized logits $z$. The sigmoid activation is applied during inference: $\sigma(z) = \frac{1}{1 + e^{-z}}$.

---

## 4. Loss Function Formulation

Nuclei occupy only 20–27% of the image canvas. Optimizing standard Cross-Entropy alone leads the model to over-predict background (achieving 75% accuracy trivially while missing nuclear boundaries).

We utilize a hybrid compound loss function:

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

### Binary Cross-Entropy with Logits
Penalizes per-pixel classification errors with numerical stability:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right]$$

### Soft Dice Loss
Directly maximizes region overlap (differentiable surrogate of the F1-score):
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}$$

Where $p_i = \sigma(z_i) \in [0, 1]$, $y_i \in \{0, 1\}$, and $\epsilon = 10^{-6}$ prevents division by zero.

---

## 5. Measured Experimental Results

All metrics below are **actual measured results** produced by `src/evaluate.py` across the independent validation and test partitions:

| Evaluation Split | Samples | Dice Coefficient (F1) | IoU (Jaccard Index) | Precision | Recall (Sensitivity) | Pixel Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Set** | 128 | **0.7857 ± 0.070** | **0.6520 ± 0.087** | 0.7400 ± 0.093 | 0.8605 ± 0.110 | **0.8880 ± 0.039** |
| **Independent Test Set** | 224 | **0.7502 ± 0.115** | **0.6116 ± 0.125** | 0.6737 ± 0.146 | **0.8842 ± 0.074** | **0.8755 ± 0.064** |

### Convergence Characteristics
- **Training Time**: 13.39 minutes for 8 epochs on CPU (AMD Ryzen 5 PRO, 6 threads).
- **Validation Dice Progression**: $0.7008 \to 0.7756 \to 0.7806 \to \mathbf{0.7951}$ (peak at epoch 6).
- **High Sensitivity**: Recall exceeds 88% on unseen patient tissue, ensuring minimal false negatives for tiny or faint chromatin regions.

---

## 6. High-Resolution Tiled Inference with 2D Hann Window Blending

Histopathology whole-slide images exceed standard GPU memory limits. Resizing a $1000 \times 1000$ or $10,000 \times 10,000$ image to $256 \times 256$ destroys cellular features.

Our sliding-window inference pipeline (`src/inference.py`):
1. Traverses the high-resolution canvas with configurable tile size ($256 \times 256$) and overlap ($64\text{ px}$).
2. Computes tile probabilities in batched tensors.
3. Applies a **2D Hann weighting window**:
   $$w(x, y) = \sin^2\left(\frac{\pi x}{L}\right) \sin^2\left(\frac{\pi y}{L}\right)$$
   This tapers probabilities toward tile boundaries, completely eliminating visible seam lines and grid boundary artifacts upon mask reconstruction.

### Benchmark on Full $1000 \times 1000$ Whole-Slide Tissue Sections:
| Metric | Measurement (AMD Ryzen CPU) |
| :--- | :--- |
| **Image Resolution** | $1000 \times 1000$ pixels (RGB) |
| **Tile Size & Overlap** | $256 \times 256$ pixels (64 px overlap) |
| **Total Tiles Processed** | 25 tiles |
| **Inference Runtime** | **1.42 seconds** |
| **Detected Nuclei Count** | 283 – 408 nuclei per tissue section |

---

## 7. Morphological Nuclei Quantification

Semantic segmentation outputs pixel classes. The quantification module (`src/inference.py`) bridges semantic segmentation to digital pathology analytics:
- **Connected Components Analysis**: Evaluates 8-connectivity clusters.
- **Morphometry Features**: Centroid $(x, y)$, pixel area, physical area ($\mu\text{m}^2$ using $0.25\ \mu\text{m/px}$ calibration), bounding box dimensions.
- **Noise Rejection**: Filters spurious debris ($< 15\text{ px}$) and macroscopic fold artifacts ($> 3500\text{ px}$).
- **Biological Caveat**: Distinguishes semantic segmentation from instance segmentation; adjacent nuclei in tight contact merge into single connected components unless resolved by marker-controlled watershed or HoVer-Net distance transforms.

---

## 8. Repository Structure

```
medical-image-segmentation/
├── data/
│   ├── raw/                  # Downloaded MoNuSeg ZIPs and extracted files
│   ├── train/                # 464 training image & mask patches
│   ├── val/                  # 128 validation image & mask patches
│   ├── test/                 # 224 independent test image & mask patches
│   └── dataset_summary.json  # Comprehensive data partition metadata
├── notebooks/
│   └── exploration.ipynb     # Exploratory analysis (stain profiles, class balance)
├── src/
│   ├── prepare_data.py       # XML parsing, patient-level split, patch extraction
│   ├── dataset.py            # PyTorch Dataset, augmentations, DataLoaders
│   ├── model.py              # Modular U-Net built from scratch in PyTorch
│   ├── train.py              # Reproducible training pipeline with checkpointing
│   ├── evaluate.py           # Dice, IoU, confusion matrix, and CSV reporting
│   ├── predict.py            # Single-image inference and 5-panel error visualization
│   └── inference.py          # High-res sliding-window tiling and quantification
├── tests/
│   ├── test_dataset.py       # Dataset loading, shapes, normalization, XML tests
│   ├── test_model.py         # Forward pass, shapes, bounds, gradient backprop
│   ├── test_metrics.py       # Dice, IoU, empty mask edge cases, loss gradients
│   └── test_inference.py     # 2D window blending, coordinate coverage, quantification
├── outputs/
│   ├── predictions/          # High-res diagnostic figures and morphometry CSVs
│   ├── plots/                # Learning curves and training history CSV
│   ├── val_metrics.csv       # Per-sample metrics across validation set
│   └── test_metrics.csv      # Per-sample metrics across independent test set
├── checkpoints/
│   ├── best_model.pth        # Top-performing checkpoint based on validation Dice
│   └── last_model.pth        # Final epoch checkpoint
├── app.py                    # Interactive Streamlit digital pathology web app
├── requirements.txt          # Pinned project dependencies
├── .gitignore                # Git exclusion rules
└── README.md                 # Project documentation
```

---

## 9. Installation & Quick Start

### Local Setup (Windows / Linux / macOS)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/medical-image-segmentation.git
   cd medical-image-segmentation
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Unit Tests**:
   ```bash
   pytest -v tests/
   ```

---

## 10. Execution Commands

### 1. Data Acquisition & Preprocessing
To parse XML annotations, partition at patient level, and generate patches:
```bash
python src/prepare_data.py
```

### 2. Model Training
Train U-Net with combined BCE + Dice loss and learning rate scheduling:
```bash
python src/train.py --epochs 8 --batch_size 8 --init_features 16 --bilinear
```

### 3. Quantitative Evaluation
Evaluate on the validation or independent test partition:
```bash
python src/evaluate.py --split val
python src/evaluate.py --split test
```

### 4. Single-Image Prediction & Diagnostic Visualization
Generate 5-panel error analysis (Image, Ground Truth, Prediction, Overlay, Error Map):
```bash
python src/predict.py --image data/val/images/sample.png --mask data/val/masks/sample.png
```

### 5. High-Resolution Tiled Inference on Whole-Slide Section
Run sliding-window inference with 2D Hann blending and cellular quantification:
```bash
python src/inference.py --image data/val/full_images/TCGA-49-4488-01Z-00-DX1.png --mask data/val/full_masks/TCGA-49-4488-01Z-00-DX1.png
```

### 6. Interactive Streamlit Web Application
Launch the interactive web UI:
```bash
streamlit run app.py
```

---

## 11. Known Limitations & Future Work

1. **Semantic vs. Instance Segmentation**: U-Net produces pixel-wise semantic masks. Touching or overlapping nuclei in dense clusters appear conjoined. Incorporating distance maps (HoVer-Net) or marker-controlled watershed post-processing would enable instance separation.
2. **Stain Normalization**: H&E slide staining varies significantly across pathology labs. Implementing Macenko or Vahadane stain normalization in the preprocessing pipeline would further improve cross-hospital generalization.
3. **Whole-Slide WSI Formats**: Currently supports standard multi-megapixel TIFF and PNG files. Integrating OpenSlide or cuCIM would allow direct tiled streaming of multi-gigabyte SVS/NDPI files.

---

## 12. Citations & References

- **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
- **MoNuSeg Dataset**: Kumar, N., Verma, R., Sharma, S., Bhargava, S., Vahadane, A., & Sethi, A. (2017). *A Dataset and a Technique for Generalized Nuclear Segmentation for Histology Images*. IEEE Transactions on Medical Imaging, 36(7), 1550-1560.
- **Grand Challenge**: [https://monuseg.grand-challenge.org/](https://monuseg.grand-challenge.org/)
- **TCGA**: The Cancer Genome Atlas research network ([https://www.cancer.gov/tcga](https://www.cancer.gov/tcga)).

---

## 13. License

Distributed under the MIT License. See `LICENSE` for details.

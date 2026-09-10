<div align="center">

# 🔎 DeepFake Noise + Wavelet + ML

### Image Forensics via Spatial Noise Statistics and Wavelet-Domain Features

*A classical, interpretable, feature-engineered alternative to end-to-end CNN deepfake detectors.*

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://deepfake-noise-wavelet-ml-hd.streamlit.app/)
[![API](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://deepfake-noise-wavelet-ml.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-LogReg-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](#)
[![PyWavelets](https://img.shields.io/badge/PyWavelets-db2%20%C2%B7%20L3-4B8BBE?style=for-the-badge)](#)

**[Live Demo](https://deepfake-noise-wavelet-ml-hd.streamlit.app/) · [API](https://deepfake-noise-wavelet-ml.onrender.com/) · [Repository](https://github.com/Harshithpatali/deepfake-noise-wavelet-ml)**

</div>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Research Objective](#2-research-objective)
3. [Dataset](#3-dataset)
4. [Why Noise Is a Forensic Signal](#4-why-noise-is-a-forensic-signal)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Feature Engineering — Mathematical Formulation](#6-feature-engineering--mathematical-formulation)
7. [Feature-Set Comparison](#7-feature-set-comparison)
8. [Model — Regularized Logistic Regression](#8-model--regularized-logistic-regression)
9. [Threshold Selection](#9-threshold-selection)
10. [Validation Set Performance](#10-validation-set-performance)
11. [Validation / Test / Holdout Protocol](#11-validation--test--holdout-protocol)
12. [Production Architecture](#12-production-architecture)
13. [API Reference](#13-api-reference)
14. [Repository Structure](#14-repository-structure)
15. [Local Development](#15-local-development)
16. [Deployment](#16-deployment)
17. [Scientific Design Principles](#17-scientific-design-principles)
18. [Limitations](#18-limitations)
19. [Future Research](#19-future-research)
20. [Disclaimer](#20-disclaimer)

---

## 1. Overview

This project investigates whether **forensic traces in image noise and wavelet representations** can distinguish authentic face images from AI-generated or manipulated ones — **without** an end-to-end convolutional network.

```text
Input Face Image
       │
       ▼
Canonical Preprocessing
       │
       ├────────────────┐
       ▼                ▼
Spatial / Noise       Wavelet
  Features            Features
       │                │
       └───────┬────────┘
               ▼
      Combined Feature Vector  (ℝ¹⁰⁵)
               │
               ▼
       L1 Feature Selection
               │
               ▼
       Logistic Regression
               │
               ▼
        Frozen Threshold τ
               │
          ┌────┴────┐
          ▼         ▼
        REAL       FAKE
```

The hypothesis: synthetic image pipelines perturb high-frequency structure and multi-scale statistics in ways that are hard to *see*, but easy to *measure*.

---

## 2. Research Objective

> **Can measurable differences in image noise and multi-scale wavelet statistics provide useful forensic evidence for detecting synthetic face images?**

A secondary objective compares the discriminative value of three feature sets:

```text
F_spatial      vs.      F_wavelet      vs.      F_combined = F_spatial ∪ F_wavelet
```

The design is deliberately **interpretable** — every dimension of the feature vector has a closed-form statistical definition, so the classifier's decision boundary is traceable back to measurable image properties rather than opaque learned filters.

---

## 3. Dataset

**DeepFakeFusion-304K** — 304,154 labeled face images drawn from multiple sources and manipulation categories.

| Split | Real | Fake | Total | Fake ratio |
|---|---:|---:|---:|---:|
| Train | 102,358 | 128,003 | 230,361 | 55.6% |
| Validation | 19,853 | 28,436 | 48,289 | 58.9% |
| Test | 11,199 | 14,305 | 25,504 | 56.1% |
| **Total** | **143,410** | **170,744** | **304,154** | **56.1%** |

Label convention: `y ∈ {0, 1}`, where `REAL = 0` and `FAKE = 1`.

---

## 4. Why Noise Is a Forensic Signal

Generation and manipulation pipelines can alter, often imperceptibly:

- high-frequency texture and residual noise
- local pixel statistics and gradients
- edge and Laplacian response
- the relationship between low- and high-frequency energy

Rather than assuming a single universal "fingerprint," the project tests whether **statistical descriptors** of these effects carry predictive signal across a large, source-diverse dataset.

---

## 5. Preprocessing Pipeline

Every image passes through an identical, fixed pipeline — no per-image tuning, no test-time augmentation:

```text
Original Image
      │
      ▼
EXIF Orientation Correction
      │
      ▼
RGB → Grayscale
      │
      ▼
Resize → 256 × 256
      │
      ▼
Float64 Representation
      │
      ▼
Feature Extraction  (105-D)
```

Grayscale conversion is intentional: the analysis targets spatial intensity structure, gradients, Laplacian response, noise residuals, and wavelet coefficients — not color.

---

## 6. Feature Engineering — Mathematical Formulation

```text
16 (spatial)  +  71 (wavelet)  +  18 (forensic descriptors)  =  105 total features
```

Let `I` be the preprocessed 256×256 grayscale image, flattened to pixel intensities `{x_1, ..., x_N}`, where `N = 65,536`.

### 6.1 Spatial / Intensity Statistics (7)

```text
mean            μ  = (1/N) Σ xᵢ
std. dev.       σ  = √[ (1/N) Σ (xᵢ − μ)² ]
variance        σ²
energy          E  = Σ xᵢ²
MAD             (1/N) Σ |xᵢ − μ|

skewness        γ₁ = [ (1/N) Σ (xᵢ − μ)³ ] / σ³
kurtosis        γ₂ = [ (1/N) Σ (xᵢ − μ)⁴ ] / σ⁴  − 3
```

→ `pixel_mean · pixel_std · pixel_variance · pixel_energy · pixel_mad · pixel_skewness · pixel_kurtosis`

### 6.2 Gradient Statistics (3)

Using Sobel operators `Gx`, `Gy`, gradient magnitude:

```text
|∇I| = √(Gx² + Gy²)
```

Mean, standard deviation, and energy of `|∇I|` → `gradient_mean · gradient_std · gradient_energy`

### 6.3 Laplacian Statistics (3)

```text
∇²I = ∂²I/∂x² + ∂²I/∂y²
```

Mean, standard deviation, and energy of `∇²I` → `laplacian_mean · laplacian_std · laplacian_energy`

### 6.4 High-Frequency Noise Residual (6)

Residual after Gaussian smoothing `I_σ = I * Gaussian(σ)`:

```text
R = I − I_σ

noise_hf_mean         = mean(R)
noise_hf_std          = std(R)
noise_hf_energy       = Σ R²

noise_hf_median_abs   = median(|R|)
noise_hf_p95_abs      = 95th percentile(|R|)
noise_hf_p99_abs      = 99th percentile(|R|)
```

### 6.5 Distributional Descriptors — Gradient & Laplacian (8)

50th / 90th / 95th / 99th percentiles of `|∇I|` and `|∇²I|`:

`gradient_p50/90/95/99_abs` · `laplacian_p50/90/95/99_abs`

### 6.6 Wavelet Decomposition (71 features)

A 2-D discrete wavelet transform (`pywt.wavedec2`) with wavelet **db2** at **level 3**:

```text
I  →  wavedec2  →  { LL3, LH3, HL3, HH3, LH2, HL2, HH2, LH1, HL1, HH1 }
```

10 sub-bands, each yielding the same 7 statistics as §6.1 (mean, std, variance, energy, MAD, skewness, kurtosis):

```text
10 bands × 7 statistics = 70 features
```

Plus a robust noise-scale estimator (Donoho's MAD estimator) over the finest detail coefficients `d`:

```text
σ̂ = median(|d|) / 0.6745

70 + 1 (σ̂) = 71 wavelet features
```

### 6.7 Wavelet Relationship Descriptors (3)

Let `E_k` be the energy of sub-band `k`, and `p_k = E_k / Σ E_j` its normalized share:

```text
wavelet_detail_energy_ratio    = ( Σ E_k for k ≠ LL3 ) / ( Σ E_k for all k )
wavelet_high_low_energy_ratio  = ( Σ E_k for k ∈ {LH, HL, HH} ) / E_LL3

wavelet_entropy                = − Σ p_k · log₂(p_k)
```

### 6.8 Local Texture Statistics (4)

For a sliding window of local standard deviations `S = {σ_w}` computed over local patches `w`:

```text
local_std_mean = mean(S)
local_std_std  = std(S)
local_std_p90  = 90th percentile(S)
local_std_p95  = 95th percentile(S)
```

---

## 7. Feature-Set Comparison

Experiments compared classifiers trained on `F_spatial`, `F_wavelet`, and `F_combined`.

> **Finding:** the wavelet/noise representation performed **slightly** better than the spatial/noise representation alone — not a dramatic margin, but a consistent one.

This supports the narrower, reproducible claim:

> *On this dataset and pipeline, wavelet/noise features provide a small but measurable advantage over spatial features alone.*

No claim is made that wavelets are universally superior across all deepfake-generation families.

---

## 8. Model — Regularized Logistic Regression

The production classifier is chosen deliberately for interpretability, small footprint, and CPU-only inference — not raw accuracy ceiling.

### 8.1 L1 Feature Selection

Feature selection is fit **on training data only**, using an L1-penalized logistic regression to induce sparsity:

```text
ŵ = argmin_w  [  −(1/N) Σᵢ ( yᵢ·log(pᵢ) + (1−yᵢ)·log(1−pᵢ) )   +   λ·‖w‖₁  ]
                └──────────────── cross-entropy loss ────────────────┘
```

Features with `ŵ_j = 0` are dropped; the surviving indices are frozen into `feature_schema_l1.json`.

### 8.2 Final Classifier

```text
pᵢ = P(FAKE | xᵢ) = sigmoid(wᵀxᵢ + b) = 1 / (1 + e^−(wᵀxᵢ + b))
```

fit on the L1-selected feature subset via training-only cross-validation over the regularization strength λ (equivalently `C = 1/λ` in scikit-learn's parameterization).

---

## 9. Threshold Selection

Rather than assuming a threshold of 0.5, the classification threshold `τ` is tuned on the **validation** set and then **frozen**:

```text
ŷᵢ = FAKE   if  pᵢ ≥ τ
     REAL   if  pᵢ < τ
```

The frozen `τ` is stored alongside the model schema so production inference never re-derives it.

---

## 10. Validation Set Performance

Metrics below are computed on the **validation split** (48,289 images) at the frozen operating threshold `τ = 0.32`, selected as described in §9.

| Metric | Value |
|---|---:|
| Threshold (τ) | 0.320 |
| ROC-AUC | 0.9249 |
| PR-AUC | 0.9124 |
| Accuracy | 0.9086 |
| Precision | 0.9062 |
| Recall (Sensitivity) | 0.9163 |
| F1-score | 0.9140 |
| Balanced Accuracy | 0.8964 |

**Confusion matrix**

| | Predicted REAL | Predicted FAKE |
|---|---:|---:|
| **Actual REAL** | TN = 6,299 | FP = 1,554 |
| **Actual FAKE** | FN = 381 | TP = 40,055 |

```text
Specificity (TNR) = TN / (TN + FP) = 6,299 / 7,853  ≈ 0.802
NPV               = TN / (TN + FN) = 6,299 / 6,680  ≈ 0.943
```

At `τ = 0.32`, the model favors **recall over precision on the FAKE class** — it is tuned to minimize missed synthetic images (FN = 381) at the cost of a higher false-positive rate on real images (FP = 1,554). This is a deliberate trade-off for a screening tool, where a missed deepfake is typically more costly than a false alarm on a genuine image.

These are validation-set numbers used to select and freeze `τ`; the locked test set and blind holdout (§11) provide the unbiased generalization check.

---

## 11. Validation / Test / Holdout Protocol

| Split | Role |
|---|---|
| **Train** | feature selection (L1), model fitting, CV hyperparameter search |
| **Validation** | model comparison, threshold τ selection |
| **Official test** | single locked evaluation, touched once |
| **Blind holdout** | independent sanity check after the model is frozen |

Repeated test-set use during development is avoided deliberately — it produces optimistic, non-generalizing performance estimates.

---

## 12. Production Architecture

```text
                         USER
                          │
                          ▼
           ┌───────────────────────────┐
           │ Streamlit Community Cloud │
           │   frontend/app.py         │
           └──────────────┬────────────┘
                          │  HTTP POST /predict
                          ▼
           ┌───────────────────────────┐
           │ Render — FastAPI Backend  │
           │   backend/main.py         │
           └──────────────┬────────────┘
                          ▼
           ┌───────────────────────────┐
           │ production_inference.py   │
           └──────────────┬────────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
     model_frozen.joblib   feature_schema_l1.json
                │                   │
                └─────────┬─────────┘
                          ▼
                     Prediction
```

The deployed system performs **inference only** — no retraining, no re-selection of features, no re-tuning of the threshold.

---

## 13. API Reference

**Base URL:** `https://deepfake-noise-wavelet-ml.onrender.com`

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /predict`

Multipart image upload → forensic prediction.

```json
{
  "prediction": "FAKE",
  "fake_probability": 0.91,
  "real_probability": 0.09,
  "threshold": 0.50,
  "representation": "combined",
  "model": "logistic_regression",
  "n_features": 42
}
```

---

## 14. Repository Structure

```text
deepfake-noise-wavelet-ml/
│
├── backend/
│   └── main.py                      # FastAPI inference service
│
├── frontend/
│   └── app.py                       # Streamlit forensic UI
│
├── src/
│   ├── data_utils.py
│   ├── features.py                  # authoritative 105-D feature extractor
│   └── modeling.py
│
├── models/
│   ├── model_frozen.joblib          # frozen Logistic Regression
│   └── feature_schema_l1.json       # selected features + threshold
│
├── notebooks/
│   ├── 01_dataset_audit
│   ├── 02_clean_and_verify_dataset
│   ├── 03_verify_locked_splits_and_holdout
│   ├── 04_extract_normal_noise_features
│   ├── 05_extract_wavelet_features
│   ├── 05b_extract_combined_features
│   ├── 06_compare_feature_sets_and_models
│   ├── 07_threshold_tuning_and_freeze
│   ├── 08_locked_test_evaluation
│   └── 09_blind_holdout_50_evaluation
│
├── metrics/
├── production_inference.py
├── requirements.txt
├── Procfile
├── .gitignore
└── README.md
```

---

## 15. Local Development

```bash
git clone https://github.com/Harshithpatali/deepfake-noise-wavelet-ml.git
cd deepfake-noise-wavelet-ml

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell

pip install -r requirements.txt
```

**Run the backend:**

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

- API → `http://127.0.0.1:8000`
- Swagger UI → `http://127.0.0.1:8000/docs`

**Run the frontend** (new terminal):

```bash
streamlit run frontend/app.py
```

The frontend targets `http://127.0.0.1:8000` locally, or `API_URL` in production.

---

## 16. Deployment

| Service | Platform | Entry point |
|---|---|---|
| Backend | [Render](https://render.com) | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Frontend | [Streamlit Community Cloud](https://streamlit.io/cloud) | `frontend/app.py` |

The frontend communicates with the backend via the `API_URL` environment variable, allowing the same code to run locally and in production.

---

## 17. Scientific Design Principles

| # | Principle |
|---|---|
| 1 | **Fixed preprocessing** — identical pipeline for every image |
| 2 | **Training-only feature selection** — test set never touched during L1 selection |
| 3 | **Training-only model fitting** — no validation/test leakage into weights |
| 4 | **Validation-based threshold** — τ tuned on validation, then frozen |
| 5 | **Locked test evaluation** — official test set used exactly once |
| 6 | **Blind holdout** — independent post-freeze sanity check |
| 7 | **Reproducible feature schema** — explicit, versioned, stored schema |
| 8 | **No production retraining** — deployed system performs inference only |

---

## 18. Limitations

- **Dataset dependence** — performance reflects the training distribution; unseen generators may behave differently.
- **Compression sensitivity** — recompression, screenshots, and resizing can alter high-frequency statistics the model relies on.
- **Information loss** — grayscale conversion and 256×256 resizing discard information that cannot be recovered.
- **Generalization** — strong test-set performance does not guarantee performance on future, unseen generation methods.
- **Errors are possible** — both false positives and false negatives occur; no binary classifier is perfect.
- **Probabilistic, not certain** — the output is a calibrated *score*, not a statement of certainty about provenance.
- **Not forensic-legal evidence** — intended as a research/screening tool only.

---

## 19. Future Research

- **Cross-generator evaluation** — train on one manipulation family, evaluate on unseen ones.
- **Cross-dataset evaluation** — evaluate on an independent dataset beyond DeepFakeFusion-304K.
- **Feature ablation** — systematically compare spatial-only, wavelet-only, noise-only, and combined subsets.
- **Frequency-domain comparison** — Fourier features, DCT features, high-pass residuals, multi-scale Laplacian pyramids.
- **Robustness testing** — JPEG recompression, cropping, blur, screenshot capture, brightness/contrast shifts.
- **Explainability** — attribution of classifier decisions back to individual forensic features.
- **Calibration analysis** — verifying that pᵢ behaves as a well-calibrated probability, not just a ranking score.
- **Model ablation** — Random Forest, Gradient Boosting, XGBoost, LightGBM, shallow MLPs, under the same strict train/val/test protocol.

---

## 20. Disclaimer

This project is intended for **research, experimentation, education, and image-forensics screening**.

A `FAKE` prediction does not establish *who* created an image, *how* it was created, or whether it was intentionally manipulated. A `REAL` prediction does not prove authenticity. The system should be treated as a machine-learning classification tool — not definitive proof of image provenance.

<div align="center">

---

**[Live Demo](https://deepfake-noise-wavelet-ml-hd.streamlit.app/) · [API](https://deepfake-noise-wavelet-ml.onrender.com/) · [GitHub Repository](https://github.com/Harshithpatali/deepfake-noise-wavelet-ml)**

</div>

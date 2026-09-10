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
10. [Validation / Test / Holdout Protocol](#10-validation--test--holdout-protocol)
11. [Production Architecture](#11-production-architecture)
12. [API Reference](#12-api-reference)
13. [Repository Structure](#13-repository-structure)
14. [Local Development](#14-local-development)
15. [Deployment](#15-deployment)
16. [Scientific Design Principles](#16-scientific-design-principles)
17. [Limitations](#17-limitations)
18. [Future Research](#18-future-research)
19. [Disclaimer](#19-disclaimer)

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

A secondary objective compares the discriminative value of:

$$
\mathcal{F}_{\text{spatial}} \quad\text{vs.}\quad \mathcal{F}_{\text{wavelet}} \quad\text{vs.}\quad \mathcal{F}_{\text{combined}} = \mathcal{F}_{\text{spatial}} \cup \mathcal{F}_{\text{wavelet}}
$$

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

Label convention: $y \in \{0, 1\}$, where $\text{REAL} = 0$, $\text{FAKE} = 1$.

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

$$
\underbrace{16}_{\text{spatial}} \;+\; \underbrace{71}_{\text{wavelet}} \;+\; \underbrace{18}_{\text{forensic descriptors}} \;=\; \underbrace{105}_{\text{total features}}
$$

Let $I \in \mathbb{R}^{256 \times 256}$ be the preprocessed grayscale image, flattened to pixel intensities $\{x_i\}_{i=1}^{N}$, $N = 65{,}536$.

### 6.1 Spatial / Intensity Statistics (7)

$$
\mu = \frac{1}{N}\sum_i x_i \qquad
\sigma = \sqrt{\frac{1}{N}\sum_i (x_i-\mu)^2} \qquad
\sigma^2 \qquad
E = \sum_i x_i^2
$$

$$
\text{MAD} = \frac{1}{N}\sum_i |x_i - \mu| \qquad
\gamma_1 = \frac{\frac{1}{N}\sum_i (x_i-\mu)^3}{\sigma^3} \qquad
\gamma_2 = \frac{\frac{1}{N}\sum_i (x_i-\mu)^4}{\sigma^4} - 3
$$

→ `pixel_mean · pixel_std · pixel_variance · pixel_energy · pixel_mad · pixel_skewness · pixel_kurtosis`

### 6.2 Gradient Statistics (3)

Using Sobel operators $G_x, G_y$, gradient magnitude:

$$
|\nabla I| = \sqrt{G_x^2 + G_y^2}
$$

Mean, standard deviation, and energy of $|\nabla I|$ →  `gradient_mean · gradient_std · gradient_energy`

### 6.3 Laplacian Statistics (3)

$$
\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}
$$

Mean, standard deviation, and energy of $\nabla^2 I$ → `laplacian_mean · laplacian_std · laplacian_energy`

### 6.4 High-Frequency Noise Residual (6)

Residual after Gaussian smoothing $I_\sigma = I * G_\sigma$:

$$
R = I - I_\sigma
$$

$$
\text{noise\_hf\_mean} = \bar R \qquad
\text{noise\_hf\_std} = \sigma_R \qquad
\text{noise\_hf\_energy} = \sum R^2
$$

$$
\text{noise\_hf\_median\_abs} = \operatorname{median}(|R|) \qquad
\text{noise\_hf\_p95\_abs} = Q_{0.95}(|R|) \qquad
\text{noise\_hf\_p99\_abs} = Q_{0.99}(|R|)
$$

### 6.5 Distributional Descriptors — Gradient & Laplacian (8)

Percentiles $Q_{0.50}, Q_{0.90}, Q_{0.95}, Q_{0.99}$ of $|\nabla I|$ and $|\nabla^2 I|$:

`gradient_p50/90/95/99_abs` · `laplacian_p50/90/95/99_abs`

### 6.6 Wavelet Decomposition (71 features)

A 2-D discrete wavelet transform (`pywt.wavedec2`) with wavelet **db2** at **level 3**:

$$
I \;\xrightarrow{\;\text{wavedec2}\;}\; \{ LL_3,\; LH_3, HL_3, HH_3,\; LH_2, HL_2, HH_2,\; LH_1, HL_1, HH_1 \}
$$

10 sub-bands, each yielding the same 7 statistics as §6.1 ($\mu, \sigma, \sigma^2, E, \text{MAD}, \gamma_1, \gamma_2$):

$$
10 \text{ bands} \times 7 \text{ statistics} = 70 \text{ features}
$$

Plus a robust noise-scale estimator (Donoho's MAD estimator) over the finest detail coefficients $d$:

$$
\hat\sigma = \frac{\operatorname{median}(|d|)}{0.6745}
$$

$$
70 + 1\;(\hat\sigma) = 71 \text{ wavelet features}
$$

### 6.7 Wavelet Relationship Descriptors (3)

Let $E_k$ be the energy of sub-band $k$, and $p_k = E_k / \sum_j E_j$ its normalized share:

$$
\text{wavelet\_detail\_energy\_ratio} = \frac{\sum_{k \ne LL_3} E_k}{\sum_k E_k}
\qquad
\text{wavelet\_high\_low\_energy\_ratio} = \frac{\sum_{k \in \{LH,HL,HH\}} E_k}{E_{LL_3}}
$$

$$
\text{wavelet\_entropy} = -\sum_k p_k \log_2 p_k
$$

### 6.8 Local Texture Statistics (4)

For a sliding window of local standard deviations $S = \{\sigma_w\}$ computed over local patches $w$:

$$
\text{local\_std\_mean} = \bar S \qquad
\text{local\_std\_std} = \sigma_S \qquad
\text{local\_std\_p90} = Q_{0.90}(S) \qquad
\text{local\_std\_p95} = Q_{0.95}(S)
$$

---

## 7. Feature-Set Comparison

Experiments compared classifiers trained on $\mathcal{F}_{\text{spatial}}$, $\mathcal{F}_{\text{wavelet}}$, and $\mathcal{F}_{\text{combined}}$.

> **Finding:** the wavelet/noise representation performed **slightly** better than the spatial/noise representation alone — not a dramatic margin, but a consistent one.

This supports the narrower, reproducible claim:

> *On this dataset and pipeline, wavelet/noise features provide a small but measurable advantage over spatial features alone.*

No claim is made that wavelets are universally superior across all deepfake-generation families.

---

## 8. Model — Regularized Logistic Regression

The production classifier is chosen deliberately for interpretability, small footprint, and CPU-only inference — not raw accuracy ceiling.

### 8.1 L1 Feature Selection

Feature selection is fit **on training data only**, using an $\ell_1$-penalized logistic regression to induce sparsity:

$$
\hat w = \arg\min_w \; \underbrace{-\frac{1}{N}\sum_{i=1}^N \Big[ y_i \log p_i + (1-y_i)\log(1-p_i) \Big]}_{\text{cross-entropy loss}} \;+\; \lambda \lVert w \rVert_1
$$

Features with $\hat w_j = 0$ are dropped; the surviving indices are frozen into `feature_schema_l1.json`.

### 8.2 Final Classifier

$$
p_i = P(\text{FAKE}\mid x_i) = \sigma(w^\top x_i + b) = \frac{1}{1 + e^{-(w^\top x_i + b)}}
$$

fit on the L1-selected feature subset via training-only cross-validation over the regularization strength $\lambda$ (equivalently $C = 1/\lambda$ in scikit-learn's parameterization).

---

## 9. Threshold Selection

Rather than assuming $\tau = 0.5$, the classification threshold is tuned on the **validation** set and then **frozen**:

$$
\hat y_i =
\begin{cases}
\text{FAKE}, & p_i \ge \tau \\
\text{REAL}, & p_i < \tau
\end{cases}
$$

The frozen $\tau$ is stored alongside the model schema so production inference never re-derives it.

---

## 10. Validation / Test / Holdout Protocol

| Split | Role |
|---|---|
| **Train** | feature selection ($\ell_1$), model fitting, CV hyperparameter search |
| **Validation** | model comparison, threshold $\tau$ selection |
| **Official test** | single locked evaluation, touched once |
| **Blind holdout** | independent sanity check after the model is frozen |

Repeated test-set use during development is avoided deliberately — it produces optimistic, non-generalizing performance estimates.

---

## 11. Production Architecture

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

## 12. API Reference

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

## 13. Repository Structure

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

## 14. Local Development

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

## 15. Deployment

| Service | Platform | Entry point |
|---|---|---|
| Backend | [Render](https://render.com) | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Frontend | [Streamlit Community Cloud](https://streamlit.io/cloud) | `frontend/app.py` |

The frontend communicates with the backend via the `API_URL` environment variable, allowing the same code to run locally and in production.

---

## 16. Scientific Design Principles

| # | Principle |
|---|---|
| 1 | **Fixed preprocessing** — identical pipeline for every image |
| 2 | **Training-only feature selection** — test set never touched during $\ell_1$ selection |
| 3 | **Training-only model fitting** — no validation/test leakage into weights |
| 4 | **Validation-based threshold** — $\tau$ tuned on validation, then frozen |
| 5 | **Locked test evaluation** — official test set used exactly once |
| 6 | **Blind holdout** — independent post-freeze sanity check |
| 7 | **Reproducible feature schema** — explicit, versioned, stored schema |
| 8 | **No production retraining** — deployed system performs inference only |

---

## 17. Limitations

- **Dataset dependence** — performance reflects the training distribution; unseen generators may behave differently.
- **Compression sensitivity** — recompression, screenshots, and resizing can alter high-frequency statistics the model relies on.
- **Information loss** — grayscale conversion and $256\times256$ resizing discard information that cannot be recovered.
- **Generalization** — strong test-set performance does not guarantee performance on future, unseen generation methods.
- **Errors are possible** — both false positives and false negatives occur; no binary classifier is perfect.
- **Probabilistic, not certain** — the output is a calibrated *score*, not a statement of certainty about provenance.
- **Not forensic-legal evidence** — intended as a research/screening tool only.

---

## 18. Future Research

- **Cross-generator evaluation** — train on one manipulation family, evaluate on unseen ones.
- **Cross-dataset evaluation** — evaluate on an independent dataset beyond DeepFakeFusion-304K.
- **Feature ablation** — systematically compare spatial-only, wavelet-only, noise-only, and combined subsets.
- **Frequency-domain comparison** — Fourier features, DCT features, high-pass residuals, multi-scale Laplacian pyramids.
- **Robustness testing** — JPEG recompression, cropping, blur, screenshot capture, brightness/contrast shifts.
- **Explainability** — attribution of classifier decisions back to individual forensic features.
- **Calibration analysis** — verifying that $p_i$ behaves as a well-calibrated probability, not just a ranking score.
- **Model ablation** — Random Forest, Gradient Boosting, XGBoost, LightGBM, shallow MLPs, under the same strict train/val/test protocol.

---

## 19. Disclaimer

This project is intended for **research, experimentation, education, and image-forensics screening**.

A `FAKE` prediction does not establish *who* created an image, *how* it was created, or whether it was intentionally manipulated. A `REAL` prediction does not prove authenticity. The system should be treated as a machine-learning classification tool — not definitive proof of image provenance.

<div align="center">

---

**[Live Demo](https://deepfake-noise-wavelet-ml-hd.streamlit.app/) · [API](https://deepfake-noise-wavelet-ml.onrender.com/) · [GitHub Repository](https://github.com/Harshithpatali/deepfake-noise-wavelet-ml)**

</div>

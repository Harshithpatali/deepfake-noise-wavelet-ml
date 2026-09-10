# DeepFake Noise + Wavelet + ML Forensics

A feature-based deepfake image-forensics system that measures spatial/noise and multi-scale wavelet statistics and uses machine learning to classify face images as **REAL** or **FAKE**.

## Live Application

**Streamlit:** https://deepfake-noise-wavelet-ml-hd.streamlit.app/

**FastAPI backend:** https://deepfake-noise-wavelet-ml.onrender.com/

**Swagger API:** https://deepfake-noise-wavelet-ml.onrender.com/docs

---

## Overview

The project investigates whether measurable differences in image noise, high-frequency structure, local statistics, and wavelet coefficients can provide useful forensic evidence for distinguishing authentic images from manipulated or synthetically generated face images.

The approach is intentionally **feature-based rather than end-to-end deep learning**. The goal is to build a compact, interpretable, CPU-friendly forensic classifier while maintaining strict separation between training, validation, test, and blind holdout data.

### Core method

```text
INPUT FACE IMAGE
       |
       v
Canonical preprocessing
RGB -> grayscale -> 256 x 256
       |
       +-----------------------+
       |                       |
       v                       v
Spatial / noise          Wavelet-domain
features                 features
       |                       |
       +-----------+-----------+
                   |
                   v
          Combined 105 features
                   |
                   v
           L1 feature selection
                   |
                   v
          Logistic Regression
                   |
                   v
          Frozen threshold
                   |
             +-----+-----+
             |           |
             v           v
           REAL        FAKE
```

---

## Research Question

> Can measurable differences in image noise and multi-scale wavelet statistics provide useful information for classifying real versus synthetic/manipulated face images?

The project also compares the predictive usefulness of spatial/noise and wavelet/noise representations.

---

## Dataset

The project uses **DeepFakeFusion-304K only**.

### Dataset size

**304,154 images**

| Split | Real | Fake | Total |
|---|---:|---:|---:|
| Train | 102,358 | 128,003 | 230,361 |
| Validation | 19,853 | 28,436 | 48,289 |
| Test | 11,199 | 14,305 | 25,504 |
| **Total** | **143,410** | **170,744** | **304,154** |

Labels:

```text
REAL = 0
FAKE = 1
```

The supplied train/validation/test split is respected. A separate 50-image-per-class blind holdout is kept outside the training, validation, and official test splits.

This repository should not be mixed with previous CIFAKE, AI-art, or unrelated datasets when reproducing these experiments.

---

## Why Analyze Noise?

Synthetic generation and image manipulation can alter fine-scale image statistics. These changes may affect:

- high-frequency residuals
- gradients
- edges
- local texture
- Laplacian responses
- frequency-specific energy
- relationships between low- and high-frequency components

Some of these traces may not be obvious from visual inspection. This project therefore treats noise and high-frequency structure as measurable forensic signals.

The objective is not to assume that every deepfake has one universal noise signature. Instead, the experiments test whether statistical measurements of these signals provide useful predictive information on the selected dataset.

---

# Feature Engineering

The complete forensic representation contains **105 features**:

```text
16 spatial/noise features
+
71 wavelet features
+
18 additional forensic descriptors
=
105 total features
```

All feature extraction is implemented in `src/features.py`.

---

## Canonical Preprocessing

Every image follows the same preprocessing pipeline:

```text
Original image
     |
     v
EXIF orientation correction
     |
     v
RGB -> grayscale
     |
     v
Resize to 256 x 256
     |
     v
Float64 representation
     |
     v
Feature extraction
```

Keeping preprocessing fixed ensures that the model receives a consistent representation during both experimentation and production inference.

---

## 1. Spatial / Normal Noise Features

The spatial representation contains 16 features:

```text
pixel_mean
pixel_std
pixel_variance
pixel_energy
pixel_mad
pixel_skewness
pixel_kurtosis

gradient_mean
gradient_std
gradient_energy

laplacian_mean
laplacian_std
laplacian_energy

noise_hf_mean
noise_hf_std
noise_hf_energy
```

These describe image intensity statistics, dispersion, distribution shape, gradients, Laplacian responses, and high-frequency noise energy.

---

## 2. Wavelet Features

The project applies a 2-D discrete wavelet transform using:

```text
Wavelet: db2
Level: 3
```

The analyzed bands are:

```text
LL3
LH3
HL3
HH3
LH2
HL2
HH2
LH1
HL1
HH1
```

For each wavelet band, the following statistics are calculated:

```text
mean
std
variance
energy
mad
skewness
kurtosis
```

This produces 70 statistical wavelet features, plus a robust noise-scale estimate:

```text
sigma_estimate
```

for a total of **71 wavelet features**.

The wavelet representation provides information at multiple scales and directional detail bands that is not captured in the same way by ordinary spatial statistics.

---

## 3. Additional Forensic Descriptors

The project also extracts 18 additional descriptors.

### High-frequency noise distribution

```text
noise_hf_median_abs
noise_hf_p95_abs
noise_hf_p99_abs
```

### Gradient distribution

```text
gradient_p50_abs
gradient_p90_abs
gradient_p95_abs
gradient_p99_abs
```

### Laplacian distribution

```text
laplacian_p50_abs
laplacian_p90_abs
laplacian_p95_abs
laplacian_p99_abs
```

### Wavelet relationships

```text
wavelet_detail_energy_ratio
wavelet_high_low_energy_ratio
wavelet_entropy
```

### Local statistics

```text
local_std_mean
local_std_std
local_std_p90
local_std_p95
```

---

# Feature-Set Comparison

A central experiment compares the spatial/noise representation with the wavelet/noise representation.

### Experimental finding

The **wavelet/noise feature representation performed slightly better than the normal spatial/noise feature representation** in the evaluated experiments.

This is consistent with the project's hypothesis that multi-scale and directional frequency information can expose useful forensic signal that is not captured as effectively by basic spatial statistics alone.

The result should be interpreted narrowly:

> On this dataset and under this experimental pipeline, wavelet/noise features provided a small performance advantage over the normal spatial/noise feature representation.

This project does **not** claim that wavelets universally outperform spatial features or that the result will necessarily transfer unchanged to every deepfake generator, dataset, or image-processing pipeline.

---

# Machine Learning

The production classifier is **Logistic Regression**.

The choice is deliberate because the project prioritizes:

- compact deployment
- CPU-friendly inference
- reproducibility
- interpretable feature effects
- stable probability outputs
- small model artifacts
- straightforward production deployment

The final production model is not retrained when an image is uploaded.

---

## L1 Feature Selection

L1-regularized feature selection is performed using **training data only**.

The selected feature names and production schema are stored in:

```text
models/feature_schema_l1.json
```

The production inference pipeline uses this frozen schema so that feature ordering and selection remain consistent with training.

---

## Model Selection

Logistic Regression regularization is tuned using training-only cross-validation.

The validation set is used for model/representation comparison and threshold selection.

The official test set is kept locked until final evaluation.

This prevents the test set from becoming part of repeated model-development decisions.

---

# Threshold Selection

The classifier produces a probability for the FAKE class:

```text
P(FAKE)
```

Rather than assuming that 0.5 is automatically the best operating threshold, the project scans candidate thresholds on the validation set and freezes the selected threshold.

Production classification follows:

```text
P(FAKE) >= frozen threshold  -> FAKE
P(FAKE) <  frozen threshold  -> REAL
```

The frozen threshold is stored in:

```text
models/feature_schema_l1.json
```

---

# Evaluation Protocol

The project uses separate stages for development and evaluation.

### Training

Used for:

- model fitting
- L1 feature selection
- cross-validation-based hyperparameter selection

### Validation

Used for:

- feature-set/model comparison
- threshold selection

### Official test

Used only for:

- final locked evaluation

### Blind holdout

Used after model freezing as an additional sanity check.

This separation is designed to reduce evaluation leakage and provide a more defensible estimate of generalization.

---

# Production Inference

The production system uses the frozen model and frozen feature schema.

```text
Uploaded image
      |
      v
Canonical preprocessing
      |
      v
105 forensic features
      |
      v
Frozen selected feature schema
      |
      v
Frozen Logistic Regression
      |
      v
Fake probability
      |
      v
Frozen threshold
      |
      +----------+
      |          |
      v          v
     REAL       FAKE
```

Production inference is implemented in:

```text
production_inference.py
```

The deployed application does not perform training, feature selection, hyperparameter tuning, or threshold tuning.

---

# Production Architecture

The live system uses two separately deployed components.

```text
                    USER
                     |
                     v
        +--------------------------+
        | Streamlit Community Cloud |
        | frontend/app.py          |
        +------------+-------------+
                     |
                     | HTTP POST /predict
                     v
        +--------------------------+
        | Render                    |
        | FastAPI backend           |
        | backend/main.py           |
        +------------+-------------+
                     |
                     v
        +--------------------------+
        | production_inference.py   |
        +------------+-------------+
                     |
              +------+------+
              |             |
              v             v
     model_frozen.joblib   feature_schema_l1.json
              |             |
              +------+------+
                     v
                 Prediction
```

The Streamlit frontend sends the uploaded image to the Render API through the `API_URL` configuration.

---

# Repository Structure

```text
deepfake-noise-wavelet-ml/
|
+-- backend/
|   +-- main.py
|
+-- frontend/
|   +-- app.py
|
+-- src/
|   +-- data_utils.py
|   +-- features.py
|   +-- modeling.py
|
+-- models/
|   +-- model_frozen.joblib
|   +-- feature_schema_l1.json
|
+-- notebooks/
|   +-- 01_dataset_audit
|   +-- 02_clean_and_verify_dataset
|   +-- 03_verify_locked_splits_and_holdout
|   +-- 04_extract_normal_noise_features
|   +-- 05_extract_wavelet_features
|   +-- 05b_extract_combined_features
|   +-- 06_compare_feature_sets_and_models
|   +-- 07_threshold_tuning_and_freeze
|   +-- 08_locked_test_evaluation
|   +-- 09_blind_holdout_50_evaluation
|
+-- metrics/
|
+-- production_inference.py
+-- requirements.txt
+-- Procfile
+-- .gitignore
+-- README.md
```

---

# Main Components

## `src/features.py`

Authoritative implementation of:

- image preprocessing
- spatial/noise features
- wavelet decomposition
- wavelet statistics
- additional forensic descriptors
- combined 105-feature representation

## `src/data_utils.py`

Provides dataset image iteration and real/fake labeling utilities.

## `production_inference.py`

Loads the frozen model and schema and performs production prediction.

## `models/model_frozen.joblib`

Frozen Logistic Regression production classifier.

## `models/feature_schema_l1.json`

Frozen production feature schema and threshold metadata.

## `backend/main.py`

FastAPI service exposing `/health` and `/predict`.

## `frontend/app.py`

Streamlit user interface for image upload and prediction visualization.

---

# API

## Health

```http
GET /health
```

Expected response:

```json
{"status":"ok"}
```

## Prediction

```http
POST /predict
```

Accepts an image upload and returns the production prediction.

Typical response structure:

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

The example probability, threshold, and selected-feature count above are illustrative response fields; the actual values are produced by the frozen production model.

---

# Local Development

The project was developed with **Python 3.12.10**.

Create the environment:

```bash
python -m venv .venv
```

Windows activation:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Start FastAPI

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Start Streamlit

In a second terminal:

```bash
streamlit run frontend/app.py
```

For local inference, the frontend can use:

```text
API_URL=http://127.0.0.1:8000
```

For production, `API_URL` points to the Render backend.

---

# Deployment

## Backend — Render

The FastAPI backend is deployed on Render.

Start command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Production backend:

https://deepfake-noise-wavelet-ml.onrender.com/

## Frontend — Streamlit Community Cloud

The Streamlit application is deployed from the GitHub repository with:

```text
Main file: frontend/app.py
Python: 3.12
```

The frontend uses the backend URL through the `API_URL` configuration.

Production frontend:

https://deepfake-noise-wavelet-ml-hd.streamlit.app/

---

# Limitations

## Dataset dependence

The classifier learns patterns represented in its training distribution. Performance can change when presented with unseen generators, manipulation methods, datasets, or image-processing pipelines.

## Compression and transformations

JPEG compression, resizing, screenshots, blur, cropping, and other transformations can change high-frequency image statistics and therefore affect forensic features.

## Grayscale preprocessing

The current pipeline converts images to grayscale and therefore does not use color-specific forensic information.

## Generalization

Performance on one dataset does not establish universal deepfake-detection capability.

## False positives and false negatives

The system can misclassify both real and fake images. A prediction is a model output, not proof of authenticity or manipulation.

## Probability interpretation

The displayed fake probability is the classifier's output score. It should not be interpreted as an absolute probability that an image was created by a particular generative system.

## Forensic use

This system is a research and screening tool. It should not be treated as definitive evidence for legal, investigative, attribution, or provenance decisions.

---

# Future Research

Potential extensions include:

- cross-generator evaluation
- cross-dataset evaluation
- feature ablation studies
- Fourier/DCT comparisons
- robustness testing under compression and resizing
- probability calibration
- feature importance and explainability
- comparison with tree-based models
- shallow neural models using the same forensic representation
- evaluation on completely unseen generation methods

A particularly important future experiment is **cross-dataset testing**, because it can reveal whether the learned forensic signal generalizes beyond the source distribution used for training.

---

# Scientific Interpretation

The main finding of this project is not that wavelets automatically solve deepfake detection.

The more defensible conclusion is:

> **For the DeepFakeFusion-304K dataset and the experimental pipeline implemented here, wavelet/noise features performed slightly better than the normal spatial/noise feature representation.**

This supports further investigation of multi-scale frequency-domain forensic features while leaving open the important question of how well those features generalize to unseen datasets and future generation methods.

---

# Project Status

- [x] Dataset audit
- [x] Dataset split verification
- [x] Image preprocessing
- [x] Spatial/noise feature extraction
- [x] Wavelet feature extraction
- [x] Combined forensic feature representation
- [x] Feature-set comparison
- [x] L1 feature selection
- [x] Logistic Regression model selection
- [x] Validation threshold tuning
- [x] Frozen production model
- [x] Locked test evaluation
- [x] Blind holdout evaluation
- [x] FastAPI inference API
- [x] Streamlit frontend
- [x] Render backend deployment
- [x] Streamlit Community Cloud deployment
- [x] End-to-end production inference

---

# Disclaimer

This project is intended for **research, education, experimentation, and image-forensics screening**.

A `FAKE` prediction does not establish who created an image, how it was created, or whether it was intentionally manipulated. A `REAL` prediction does not prove authenticity.

The system should therefore be interpreted as a machine-learning classification tool rather than definitive proof of image provenance.

---

## Live Demo

**DeepFake Forensics:** https://deepfake-noise-wavelet-ml-hd.streamlit.app/

**FastAPI Backend:** https://deepfake-noise-wavelet-ml.onrender.com/

**GitHub:** https://github.com/Harshithpatali/deepfake-noise-wavelet-ml

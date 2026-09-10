# Windows instructions

## 1. Install Python

Use Python 3.11.x.

Check:
    py -3.11 --version

## 2. Create the venv

Open Command Prompt and enter the project folder:

    cd /d D:\deepfake_noise_wavelet_ml

Create environment:

    py -3.11 -m venv .venv

Activate:

    .venv\Scripts\activate

You should see `(.venv)` in the terminal.

## 3. Install requirements

    python -m pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt

Register Jupyter kernel:

    python -m ipykernel install --user --name deepfake-noise-wavelet --display-name "DeepFake Noise Wavelet ML"

Verify:

    python --version
    pip list

## 4. Put the dataset in the project

Copy the complete DeepFakeFusion-304K folder here:

    data\raw\deepfake_merged_dataset\

It must contain:

    train\real
    train\fake
    val\real
    val\fake
    test\real
    test\fake
    metadata\

Do not rename the class folders.

## 5. Put the 50 holdout images here

    data\holdout_50\real\
    data\holdout_50\fake\

There must be exactly 50 total. Both classes must be present.

Do not put these 50 images anywhere inside train/val/test.

## 6. Start Jupyter

From the activated venv:

    jupyter notebook

Open the project folder and run notebooks in this exact order:

    01_dataset_audit.ipynb
    02_clean_and_verify_dataset.ipynb
    03_verify_locked_splits_and_holdout.ipynb
    04_extract_normal_noise_features.ipynb
    05_extract_wavelet_features.ipynb
    05b_extract_combined_features.ipynb
    06_compare_feature_sets_and_models.ipynb
    07_threshold_tuning_and_freeze.ipynb
    08_locked_test_evaluation.ipynb
    09_blind_holdout_50_evaluation.ipynb

Do not skip a notebook if it reports an error or data-quality problem.

## 7. Application after the model is frozen

Terminal 1:

    .venv\Scripts\activate
    uvicorn backend.main:app --host 127.0.0.1 --port 8000

Terminal 2:

    .venv\Scripts\activate
    streamlit run frontend/app.py

Then open the Streamlit address shown in the terminal.

## Important scientific rules

- Test is never used for model selection.
- Holdout is never used for tuning.
- Threshold is tuned only on validation.
- The feature extractor is shared by all splits and production inference.
- The original dataset is never overwritten.

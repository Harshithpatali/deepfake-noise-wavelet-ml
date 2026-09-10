from pathlib import Path
import sys
import json
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features import extract_all


MODEL_PATH = ROOT / "models" / "model_frozen.joblib"
SCHEMA_PATH = ROOT / "models" / "feature_schema_l1.json"


# Load the frozen production artifacts once when the backend starts.
MODEL = joblib.load(MODEL_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    SCHEMA = json.load(f)

SELECTED_FEATURES = SCHEMA["selected_features"]
THRESHOLD = float(SCHEMA["threshold"])


def predict_image(image_path):

    # Extract the same 105 forensic features used during training.
    features = extract_all(image_path)

    # Select exactly the frozen L1-selected features
    # and preserve their training order.
    X = [[features[c] for c in SELECTED_FEATURES]]

    # Frozen Logistic Regression prediction.
    fake_probability = float(
        MODEL.predict_proba(X)[0, 1]
    )

    prediction = (
        "FAKE"
        if fake_probability >= THRESHOLD
        else "REAL"
    )

    return {
        "prediction": prediction,
        "fake_probability": fake_probability,
        "real_probability": 1.0 - fake_probability,
        "threshold": THRESHOLD,
        "representation": "combined",
        "model": "logistic_regression",
        "n_features": len(SELECTED_FEATURES),
    }
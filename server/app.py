"""TrustMedX prediction API.

Serves the static dashboard and a JSON prediction endpoint backed by the
CatBoost model exported by train_model.py.

Run from the repository root:
    uvicorn server.app:app --port 8000
or from server/:
    uvicorn app:app --port 8000
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SERVER_DIR)
ARTIFACT_DIR = os.path.join(SERVER_DIR, "artifacts")

sys.path.insert(0, SERVER_DIR)
from train_model import engineer_features  # noqa: E402

with open(os.path.join(ARTIFACT_DIR, "metadata.json")) as f:
    META = json.load(f)

MODEL = CatBoostClassifier()
MODEL.load_model(os.path.join(ARTIFACT_DIR, "model.cbm"))
PREPROCESSOR = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))

T = META["temperature"]
QHAT = META["conformal_qhat"]
FEATURE_ORDER = META["feature_order"]

DISCLAIMER = ("This is a research tool output, not a clinical diagnosis. "
              "Not for clinical use.")

app = FastAPI(title="TrustMedX API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/schema")
def schema():
    """Form-building schema: fields, allowed values, and defaults."""
    return {
        "raw_input_cols": META["raw_input_cols"],
        "category_levels": META["category_levels"],
        "defaults": META["defaults"],
        "med_cols": META["med_cols"],
        "model": {
            "name": "CatBoost (feature-engineered)",
            "val_auroc": META["val_auroc"],
            "val_auprc": META["val_auprc"],
            "temperature": T,
            "conformal_qhat": QHAT,
        },
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/predict")
def predict(patient: dict):
    # start from training-set defaults, overlay whatever the form sent
    row = {}
    for col in META["raw_input_cols"]:
        row[col] = META["defaults"].get(col, "Unknown")
    for key, value in patient.items():
        if key not in row:
            continue
        row[key] = value

    df = pd.DataFrame([row])
    # coerce numerics
    for col in df.columns:
        if col in META["numeric_cols"]:
            try:
                df[col] = pd.to_numeric(df[col])
            except (TypeError, ValueError):
                raise HTTPException(400, f"Field '{col}' must be numeric.")

    try:
        fe = engineer_features(df)
        X = PREPROCESSOR.transform(fe[META["categorical_cols"] + META["numeric_cols"]])
    except Exception as exc:
        raise HTTPException(400, f"Could not process input: {exc}")

    raw_prob = float(MODEL.predict_proba(X)[0, 1])
    clipped = min(max(raw_prob, 1e-6), 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    calibrated = float(1 / (1 + np.exp(-logit / T)))

    # split-conformal prediction set at 90% coverage
    pred_set = []
    if calibrated <= QHAT:
        pred_set.append("no readmission within 30 days")
    if (1 - calibrated) <= QHAT:
        pred_set.append("readmission within 30 days")

    # per-patient SHAP attribution
    shap_vals = MODEL.get_feature_importance(data=Pool(X), type="ShapValues")[0]
    contribs = shap_vals[:-1]
    order = np.argsort(np.abs(contribs))[::-1][:6]
    top_factors = [
        {
            "feature": FEATURE_ORDER[i],
            "shap": round(float(contribs[i]), 4),
            "direction": "increases risk" if contribs[i] > 0 else "decreases risk",
        }
        for i in order
    ]

    risk_level = "HIGH" if calibrated >= 0.5 else ("MODERATE" if calibrated >= 0.3 else "LOW")

    return {
        "risk_level": risk_level,
        "raw_probability": round(raw_prob, 4),
        "calibrated_probability": round(calibrated, 4),
        "confidence_statement": (
            f"The model predicts a {calibrated:.1%} calibrated probability of "
            f"30-day readmission, classified as {risk_level} risk."
        ),
        "conformal_prediction_set": pred_set,
        "conformal_note": (
            "90% coverage guarantee: the true outcome falls inside this set "
            "for at least ~90% of patients. A two-label set means the model "
            "cannot confidently distinguish the outcome for this patient."
        ),
        "top_factors": top_factors,
        "disclaimer": DISCLAIMER,
    }


# static dashboard (mounted last so /api/* wins)
app.mount("/", StaticFiles(directory=REPO_ROOT, html=True), name="static")

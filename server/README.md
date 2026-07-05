# TrustMedX Prediction Server

FastAPI service that runs the trained CatBoost readmission model behind the interactive
form in `../predict.html`, with temperature-scaled probabilities, a 90%-coverage split
conformal prediction set, and per-patient SHAP attributions.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --port 8000
```

Then open **http://localhost:8000/predict.html** — the server hosts both the API and the
static site, so this one command gives the full experience (dashboard + interactive demo).

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/schema` | GET | Field list, category levels, defaults, model metadata (used by the form to build itself) |
| `/api/predict` | POST | JSON patient record → risk level, raw + calibrated probability, conformal set, top SHAP factors |

Fields omitted from the request fall back to training-set modes/medians, so partial
records are fine.

## Artifacts

`artifacts/` contains the trained model (`model.cbm`), the fitted preprocessing pipeline
(`preprocessor.joblib`), and `metadata.json` (feature schema, temperature T, conformal
quantile, validation metrics). These are committed, so no retraining is needed after
cloning.

To retrain from scratch (downloads the UCI Diabetes 130-US Hospitals dataset, ~2 min):

```bash
python train_model.py
```

The script reproduces the notebook pipeline: patient-level 80/10/10 GroupShuffleSplit
(seed 42), feature engineering, OrdinalEncoder + StandardScaler, CatBoost
(300 iterations, depth 6, lr 0.05, class-weighted). Reproduced validation AUROC: 0.6658
(notebook: 0.6632 — the small difference comes from UCI vs. Kaggle missing-value encoding).

## Deploying the API online

GitHub Pages serves only static files, so the form on the Pages site needs the API hosted
somewhere with Python:

- **Render** (free tier): new Web Service from this repo, build command
  `pip install -r server/requirements.txt`, start command
  `uvicorn server.app:app --host 0.0.0.0 --port $PORT`.
- **Hugging Face Spaces**: create a Space (SDK: Docker or Gradio-free custom), copy the
  `server/` folder plus `predict.html`/`index.html`/`results/`/`plots/`.

After deploying, open `predict.html` → **API settings** and paste the service URL; the
page stores it in the browser and uses it for all requests.

> **Research prototype only — not for clinical use.** See `../governance/DISCLAIMER.md`.

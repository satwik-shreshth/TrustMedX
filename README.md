![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Research Project](https://img.shields.io/badge/Type-Research%20Project-brightgreen.svg)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20GPU-red.svg)
![Status](https://img.shields.io/badge/Status-Active%20Development-yellow.svg)
![ML](https://img.shields.io/badge/ML-Clinical%20Decision%20Support-orange.svg)
![Models](https://img.shields.io/badge/Models-CatBoost%20%7C%20LightGBM%20%7C%20XGBoost-yellow.svg)

---

# 🏥 TrustMedX
### Trustworthy Clinical Decision Support — 30-Day Diabetes Readmission Prediction with Calibrated Uncertainty and Grounded Explanations

---

## 📌 Project Overview

**TrustMedX** studies a question most clinical ML work skips: a model can rank patients by
readmission risk reasonably well — but can a clinician actually *trust* an individual
prediction enough to act on it?

Two gaps usually block that trust. First, predicted probabilities are typically
miscalibrated, so a "70% risk" flag may correspond to a true frequency of 50% or 90%.
Second, feature-attribution explanations (SHAP, LIME) show *which* inputs mattered but not
whether the reasoning is grounded in any verifiable clinical evidence.

TrustMedX addresses both in one reproducible pipeline on the **Diabetes 130-US Hospitals**
dataset (101,766 encounters, 130 hospitals, 1999–2008): 14 models are compared with proper
patient-level splitting, the best model's probabilities are calibrated by temperature
scaling, split conformal prediction adds distribution-free coverage guarantees, and an
explanation engine combines per-patient SHAP attributions with retrieved clinical evidence
into structured, cited narratives — under an explicit constraint against fabricating
support. That constraint is stress-tested with a red-team probe using deliberately
irrelevant "evidence."

🔗 **Live dashboard:** https://satwik-shreshth.github.io/TrustMedX/
🔮 **Interactive demo:** `predict.html` — enter a patient encounter, get a calibrated risk
prediction with conformal set and SHAP explanation

---

## ⭐ Key Features

### 🔬 Model Benchmark

- **14 configurations** compared: logistic regression, random forest, XGBoost, CatBoost,
  LightGBM, MLP, TabNet — on raw and feature-engineered inputs — plus stacking and
  averaging ensembles
- **Patient-level grouped 80/10/10 split** so no patient's encounters leak across partitions
- Feature engineering from ICD-9 diagnosis grouping, prior-utilization totals, medication
  change counts, and stay-intensity ratios

### 🎯 Uncertainty Quantification

- **Temperature scaling** for probability calibration (ECE 0.3385 → 0.3257,
  Brier 0.2172 → 0.2156)
- **Split conformal prediction**: 89.7% empirical coverage at a 90% target, average
  prediction-set size 1.58
- **Bootstrap confidence intervals** (n=1000) on AUROC and AUPRC

### 🧠 Explainability & Grounded Explanation

- SHAP TreeExplainer for global importance and per-patient attributions
- Retrieval over an embedded clinical-evidence corpus (sentence-transformer + FAISS)
- Explanation engine that merges prediction, confidence, SHAP factors, and retrieved
  evidence into a structured, cited output
- **Red-team hallucination test**: fed fabricated, irrelevant evidence, the engine
  explicitly flags it as unsupportive instead of weaving it into a justification

### 🧪 Statistical Rigor

- DeLong's test on AUROC differences between the top three models
- McNemar's test on paired decisions at a fixed threshold
- Fairness audit across race, gender, and age subgroups

### 🔮 Interactive Prediction Service

- Patient-input form (`predict.html`) backed by a FastAPI service (`server/`)
- Returns raw + calibrated probability, risk level, 90%-coverage conformal set, and the
  top SHAP factors for that individual prediction
- Trained model artifacts committed under `server/artifacts/` — clone and run, no retraining

---

## 📊 Results

| Model | Val AUROC | Val AUPRC |
|---|---|---|
| Majority/chance | 0.500 | 0.112 |
| Logistic regression | 0.6275 | 0.1930 |
| Random forest | 0.6492 | 0.1943 |
| XGBoost | 0.6620 | 0.2215 |
| Averaging ensemble | 0.6654 | 0.2197 |
| LightGBM | 0.6671 | **0.2261** |
| **CatBoost** | **0.6672** | 0.2201 |

95% bootstrap CI for the final model AUROC: **[0.646, 0.679]**.

### Notable findings

- **Gradient-boosted trees hit a ~0.665–0.668 AUROC ceiling**, consistent with published
  benchmarks on this dataset — the limit reflects missing signal (socioeconomic status,
  post-discharge support) rather than model choice. Deep learning (MLP, TabNet) did not
  close the gap.
- **Equal ranking ≠ equal decisions.** DeLong's test finds CatBoost, LightGBM, and XGBoost
  statistically indistinguishable on AUROC (all p > 0.05), yet McNemar's test shows they
  disagree significantly (p < 0.001) on *which* patients get flagged at a 0.5 threshold —
  threshold selection deserves as much attention as headline AUROC.
- **The explanation engine refuses fabricated evidence.** Presented with a fake study about
  window-seat preferences, it responded that the evidence "does not directly support or
  refute the identified risk factors" instead of inventing a rationale.
- **Discrimination degrades for the oldest patients** (80–90: AUROC 0.616; 90–100: 0.597 vs.
  40–50: 0.745) — precisely the subgroup where accurate stratification matters most.

---

## 🚀 Getting Started

### Dashboard only (static)

```bash
python -m http.server 8000
# open http://localhost:8000
```

### Full experience with live predictions

```bash
pip install -r server/requirements.txt
cd server
uvicorn app:app --port 8000
# open http://localhost:8000/predict.html
```

### Retrain from scratch (~2 min, downloads the UCI dataset)

```bash
cd server && python train_model.py
```

### Build the paper (requires a TeX distribution)

```bash
make paper
```

---

## 📁 Repository Structure

```
├── index.html            → research dashboard (GitHub Pages)
├── predict.html          → interactive prediction form
├── server/               → FastAPI prediction service + training script
│   └── artifacts/        → trained model, preprocessor, metadata
├── trustmedx_paper.tex   → IEEE-format manuscript (+ compiled PDF)
├── TrustMedX.ipynb       → full experimental notebook
├── results/              → metrics (results.json, CSVs)
├── plots/                → generated figures
├── governance/           → data card, model card, disclaimer
```

---

## ⚠️ Known Limitations

- Predictive performance (AUROC ≈ 0.67), while consistent with the literature, is far below
  any threshold for clinical deployment
- Small race subgroups (n < 150) make several fairness estimates unstable; the subgroup
  analysis is retrospective and unadjusted
- The retrieval corpus is a small, illustrative, paraphrased set — not a comprehensive
  guideline database
- Data reflect 1999–2008 care patterns and may not generalize to current practice
- The red-team evaluation is a single adversarial probe, not a systematic robustness suite

---

## 🗺️ Roadmap

- [ ] Age-stratified modeling or elderly-specific features to close the elderly performance gap
- [ ] Expand the retrieval corpus with authoritative clinical guidelines
- [ ] Decision-curve analysis for threshold selection
- [ ] Prospective validation on a contemporary dataset
- [ ] Broader adversarial testing of the explanation engine

---

## 📜 License

This project is licensed under the **Creative Commons Attribution–NonCommercial 4.0
International (CC BY-NC 4.0)** License.

**You Are Free To:** Share, adapt, research, teach, and modify with attribution
**Terms:** Attribution required, non-commercial use only
**Legal Code:** https://creativecommons.org/licenses/by-nc/4.0/legalcode

---

## 👨‍💻 Author

**Satwik Shreshth**
**Contact:** satwikshreshth2002@gmail.com

---

## ⚠️ Disclaimer

TrustMedX is a research prototype. It is **not** a medical device, has **not** been reviewed
or approved by any regulatory body, and must **not** be used to make clinical decisions
about real patients. All predictions, explanations, and confidence scores demonstrate a
research methodology only. Any real clinical decision must be made by a licensed healthcare
professional using validated, approved clinical tools and their own judgment.

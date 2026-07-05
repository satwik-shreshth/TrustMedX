# TrustMedX

**A Trustworthy Clinical Decision Support Framework for 30-Day Diabetes Readmission Prediction**

Satwik Shreshth · satwikshreshth2002@gmail.com

> ⚠️ **Research prototype only.** TrustMedX is not a medical device, has not been reviewed or
> approved by any regulatory body, and must not be used to make real clinical decisions.
> See [`governance/DISCLAIMER.md`](governance/DISCLAIMER.md).

## What is TrustMedX?

Most clinical ML research reports accuracy in isolation. TrustMedX integrates, in one
reproducible pipeline on the Diabetes 130-US Hospitals dataset (101,766 encounters):

- **14-model comparison** — gradient-boosted trees (CatBoost best: AUROC 0.6672), neural
  networks, and ensembles, with DeLong / McNemar significance testing
- **Explainability** — SHAP global and per-patient explanations
- **Uncertainty quantification** — temperature scaling (ECE 0.3385 → 0.3257), split conformal
  prediction (89.7% empirical coverage at 90% target), bootstrap confidence intervals
- **Retrieval-augmented explanation** — FAISS retrieval + LLM synthesis of structured, cited
  explanations, with a red-team test showing the engine refuses fabricated evidence
- **Fairness analysis** — subgroup AUROC across race, gender, and age

## Repository layout

```
├── index.html            ← research dashboard (GitHub Pages site)
├── trustmedx_paper.tex   ← IEEE-format paper (IEEEtran)
├── trustmedx_paper.pdf   ← compiled paper
├── references.bib        ← bibliography (verify entries before submission)
├── Makefile              ← build automation
├── TrustMedX.ipynb       ← full experimental notebook
├── results/              ← metrics (results.json, CSVs)
├── plots/                ← generated figures (PNG)
├── governance/           ← DATA_CARD, MODEL_CARD, DISCLAIMER
└── manuscript/           ← original markdown draft
```

## Live dashboard

The dashboard is deployed on GitHub Pages:
**https://satwik-shreshth.github.io/TrustMedX/**

It is a single self-contained `index.html` that loads all numbers at runtime from
`results/results.json` — nothing is hardcoded.

### Run it locally

Browsers block `fetch()` from `file://` URLs, so serve the folder:

```bash
python -m http.server 8000
# then open http://localhost:8000
```

## Build the paper

Requires a TeX distribution (MiKTeX / TeX Live):

```bash
make paper
# or manually:
pdflatex trustmedx_paper.tex && bibtex trustmedx_paper && pdflatex trustmedx_paper.tex && pdflatex trustmedx_paper.tex
```

**Note:** entries in `references.bib` were assembled from memory — verify every author list,
venue, and page range against the original publications before submitting anywhere.

## Dataset

Diabetes 130-US Hospitals (Strack et al., 2014), 101,766 encounters, 1999–2008, de-identified,
publicly available from the UCI Machine Learning Repository.

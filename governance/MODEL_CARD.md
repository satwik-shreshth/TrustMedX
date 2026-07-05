# Model Card — TrustMedX Readmission Risk Model

## Model Details
Best-performing model: CatBoost Classifier (raw features)
Validation AUROC: 0.6672
Validation AUPRC: 0.2201
95% Bootstrap CI (AUROC): [np.float64(0.645673381418253), np.float64(0.6790846322043735)]

## Intended Use
Research prototype only. Demonstrates a trustworthy clinical decision support (CDS) framework combining ML, explainability (SHAP), uncertainty quantification (calibration, conformal prediction), and retrieval-augmented explanation.

## NOT Intended For
- Clinical deployment or real patient care decisions
- Use without a licensed clinician's review
- Any use outside research/educational demonstration

## Models Evaluated
Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM, MLP (deep learning), TabNet, Stacking Ensemble, Averaging Ensemble.
CatBoost and LightGBM performed best (~0.665-0.668 AUROC); this is consistent with published benchmarks on this dataset, reflecting genuine limits of predictive signal available in structured EHR data for this task.

## Calibration
- Expected Calibration Error (ECE) after temperature scaling: 0.3257
- Brier Score after calibration: 0.2156

## Uncertainty Quantification
- Conformal prediction empirical coverage: 0.8969 (target 90%)
- Average prediction set size: 1.5838

## Fairness / Subgroup Considerations
Not yet evaluated in this version — flagged as a limitation. Future work should assess performance across race, gender, and age subgroups given known healthcare disparities.

## Ethical Considerations
Predictions are for research demonstration only. False negatives (missed high-risk patients) and false positives (unnecessary intervention) both carry real costs; this model has NOT been validated for clinical safety or fairness at the level required for deployment.

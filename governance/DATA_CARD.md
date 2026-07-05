# Data Card — Diabetes 130-US Hospitals Dataset

## Source
UCI Machine Learning Repository / Kaggle (brandao/diabetes), originally from Strack et al. (2014).

## Description
101,766 hospital encounters for diabetic patients across 130 US hospitals (1999-2008).

## Task
Binary classification: predict whether a patient is readmitted within 30 days of discharge.

## Class Balance
- Not readmitted <30 days: 88.8%
- Readmitted <30 days: 11.2%

## Known Limitations
- Missing values in weight (96.9%), max_glu_serum (94.7%), A1Cresult (83.3%), medical_specialty (49.1%), payer_code (39.6%)
- Encounter-level data; some patients have multiple encounters (handled via patient-level split to avoid leakage)
- No socioeconomic, housing, or post-discharge care variables — known strong predictors of readmission that are absent from this dataset
- Data is from 1999-2008; clinical practice patterns have changed since

## Split Strategy
Patient-level grouped split (80% train / 10% val / 10% test) using GroupShuffleSplit on patient_nbr, to prevent the same patient appearing in multiple splits.

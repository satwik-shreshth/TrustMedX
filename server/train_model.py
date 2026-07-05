"""Reproduce the TrustMedX CatBoost (feature-engineered) model and export
serving artifacts.

Mirrors the training pipeline in TrustMedX.ipynb: UCI Diabetes 130-US
Hospitals data, patient-level 80/10/10 GroupShuffleSplit (seed 42),
feature engineering, OrdinalEncoder + StandardScaler preprocessing,
CatBoost (300 iterations, depth 6, lr 0.05, class-weighted), followed by
temperature scaling and a split-conformal quantile fitted on the
validation set.

Outputs (server/artifacts/):
    model.cbm          trained CatBoost model
    preprocessor.joblib  fitted ColumnTransformer
    metadata.json      feature schema, category levels, T, qhat, metrics
"""

import io
import json
import os
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

UCI_URL = ("https://archive.ics.uci.edu/static/public/296/"
           "diabetes+130-us+hospitals+for+years+1999-2008.zip")
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
DATA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetic_data.csv")

MED_COLS = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']


def load_data():
    if not os.path.exists(DATA_CACHE):
        print("Downloading dataset from UCI ...")
        with urllib.request.urlopen(UCI_URL) as resp:
            outer = zipfile.ZipFile(io.BytesIO(resp.read()))
        # the outer zip contains dataset_diabetes/diabetic_data.csv (possibly nested)
        csv_name = next(n for n in outer.namelist() if n.endswith("diabetic_data.csv"))
        with outer.open(csv_name) as f:
            raw = f.read()
        with open(DATA_CACHE, "wb") as f:
            f.write(raw)
    df = pd.read_csv(DATA_CACHE, dtype=str, keep_default_na=False)
    # numeric columns back to int
    int_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses']
    for c in int_cols:
        df[c] = df[c].astype(int)
    return df


def clean(df):
    df = df.drop(columns=['weight', 'payer_code', 'encounter_id'])
    # lab columns: 'None' means test not ordered
    df['max_glu_serum'] = df['max_glu_serum'].replace('None', 'Not_tested')
    df['A1Cresult'] = df['A1Cresult'].replace('None', 'Not_tested')
    for col in ['race', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].replace('', 'Unknown')
    df['target'] = (df['readmitted'] == '<30').astype(int)
    df = df.drop(columns=['readmitted'])
    return df


def map_diag_category(code):
    if pd.isna(code) or code in ('Unknown', '?'):
        return 'Unknown'
    try:
        code_num = float(code)
    except ValueError:
        if str(code).startswith('V') or str(code).startswith('E'):
            return 'Injury_External'
        return 'Other'
    if 390 <= code_num <= 459 or code_num == 785:
        return 'Circulatory'
    elif 460 <= code_num <= 519 or code_num == 786:
        return 'Respiratory'
    elif 520 <= code_num <= 579 or code_num == 787:
        return 'Digestive'
    elif 250 <= code_num < 251:
        return 'Diabetes'
    elif 800 <= code_num <= 999:
        return 'Injury'
    elif 710 <= code_num <= 739:
        return 'Musculoskeletal'
    elif 580 <= code_num <= 629 or code_num == 788:
        return 'Genitourinary'
    elif 140 <= code_num <= 239:
        return 'Neoplasm'
    return 'Other'


def engineer_features(df):
    df = df.copy()
    df['total_prior_visits'] = (df['number_outpatient'] + df['number_emergency']
                                + df['number_inpatient'])
    df['diag_1_category'] = df['diag_1'].apply(map_diag_category)
    df['diag_2_category'] = df['diag_2'].apply(map_diag_category)
    df['diag_3_category'] = df['diag_3'].apply(map_diag_category)
    df['num_med_changes'] = df[MED_COLS].apply(
        lambda row: sum(1 for v in row if v in ['Up', 'Down']), axis=1)
    df['num_meds_prescribed'] = df[MED_COLS].apply(
        lambda row: sum(1 for v in row if v != 'No'), axis=1)
    df['procedures_per_day'] = df['num_procedures'] / (df['time_in_hospital'] + 1)
    df['meds_per_day'] = df['num_medications'] / (df['time_in_hospital'] + 1)
    df['is_elderly'] = df['age'].apply(
        lambda x: 1 if x in ['[70-80)', '[80-90)', '[90-100)'] else 0)
    return df


def temperature_scale(logits, T):
    return 1 / (1 + np.exp(-logits / T))


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = clean(load_data())

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(gss.split(df, groups=df['patient_nbr']))
    train_val_df, test_df = df.iloc[train_val_idx], df.iloc[test_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df['patient_nbr']))
    train_df, val_df = train_val_df.iloc[train_idx], train_val_df.iloc[val_idx]

    def xy(d):
        d = d.drop(columns=['patient_nbr'])
        return d.drop(columns=['target']), d['target']

    X_train, y_train = xy(train_df)
    X_val, y_val = xy(val_df)
    X_test, y_test = xy(test_df)
    print(f"Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    X_train_fe = engineer_features(X_train)
    X_val_fe = engineer_features(X_val)
    X_test_fe = engineer_features(X_test)

    categorical_cols = X_train_fe.select_dtypes(include='object').columns.tolist()
    numeric_cols = [c for c in X_train_fe.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         categorical_cols),
        ('num', StandardScaler(), numeric_cols),
    ])
    X_train_p = preprocessor.fit_transform(X_train_fe)
    X_val_p = preprocessor.transform(X_val_fe)

    model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42, verbose=0)
    model.fit(X_train_p, y_train)

    val_probs = model.predict_proba(X_val_p)[:, 1]
    auroc = roc_auc_score(y_val, val_probs)
    auprc = average_precision_score(y_val, val_probs)
    print(f"Validation AUROC {auroc:.4f}  AUPRC {auprc:.4f}  "
          f"(notebook reference: 0.6632 / 0.2155)")

    # temperature scaling on validation set
    clipped = np.clip(val_probs, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    y_val_arr = y_val.values

    def nll(T):
        p = np.clip(temperature_scale(logits, T), 1e-6, 1 - 1e-6)
        return -np.mean(y_val_arr * np.log(p) + (1 - y_val_arr) * np.log(1 - p))

    optimal_T = minimize_scalar(nll, bounds=(0.1, 10), method='bounded').x
    calibrated = temperature_scale(logits, optimal_T)
    print(f"Optimal temperature T = {optimal_T:.4f} (notebook reference: 0.7494)")

    # split conformal quantile on validation set
    alpha = 0.1
    scores = np.where(y_val_arr == 1, 1 - calibrated, calibrated)
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = float(np.quantile(scores, q_level))
    print(f"Conformal qhat = {qhat:.4f} (notebook reference: 0.6684)")

    model.save_model(os.path.join(ARTIFACT_DIR, "model.cbm"))
    joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))

    # category levels for form dropdowns; medians/modes as form defaults
    cat_levels = {c: sorted(X_train_fe[c].unique().tolist()) for c in categorical_cols}
    defaults = {}
    for c in categorical_cols:
        defaults[c] = X_train_fe[c].mode().iloc[0]
    for c in numeric_cols:
        defaults[c] = float(X_train_fe[c].median())

    metadata = {
        "feature_order": categorical_cols + numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "raw_input_cols": X_train.columns.tolist(),
        "med_cols": MED_COLS,
        "category_levels": cat_levels,
        "defaults": defaults,
        "temperature": float(optimal_T),
        "conformal_qhat": qhat,
        "val_auroc": float(auroc),
        "val_auprc": float(auprc),
    }
    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Artifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()

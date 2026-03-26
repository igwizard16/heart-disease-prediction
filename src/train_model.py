"""
train_model.py
Run this ONCE before launching the app.
Usage:  python src/train_model.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score, confusion_matrix)
from imblearn.over_sampling  import SMOTE
import xgboost as xgb

DATA_PATH  = os.path.join(BASE_DIR, "data",
             "heart_disease_health_indicators_BRFSS2015.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL         = "HeartDiseaseorAttack"
NUMERICAL_FEATURES = ["BMI", "MentHlth", "PhysHlth"]


def load_and_clean(path):
    print(f"\n[1/5] Loading data ...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}\nPut the CSV inside /data/ folder.")
    df = pd.read_csv(path)
    df = df.drop_duplicates().dropna()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    counts = df[TARGET_COL].value_counts()
    print(f"      Shape: {df.shape}  |  Class 0: {counts[0]}  Class 1: {counts[1]}")
    return df


def prepare_data(df):
    print("\n[2/5] Splitting + Scaling + SMOTE ...")
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values.astype(float)
    y = df[TARGET_COL].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    num_idx = [feature_cols.index(f) for f in NUMERICAL_FEATURES if f in feature_cols]
    scaler  = StandardScaler()
    X_train[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
    X_test[:, num_idx]  = scaler.transform(X_test[:, num_idx])

    sm = SMOTE(random_state=42, k_neighbors=5)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"      Train after SMOTE: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, feature_cols


def train_all(X_train, y_train):
    print("\n[3/5] Training 3 models ...")
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=42, solver="lbfgs", C=0.1),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0),
    }
    trained = {}
    for name, mdl in models.items():
        print(f"      {name} ...", end="", flush=True)
        mdl.fit(X_train, y_train)
        trained[name] = mdl
        print(" done")
    return trained


def evaluate_all(trained, X_test, y_test):
    print("\n[4/5] Evaluation Report")
    print(f"\n{'─'*62}")
    print(f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Recall':>8} {'F1':>7}")
    print(f"{'─'*62}")
    results = []
    for name, mdl in trained.items():
        y_pred = mdl.predict(X_test)
        r = dict(name=name,
                 accuracy  = round(accuracy_score(y_test, y_pred), 4),
                 precision = round(precision_score(y_test, y_pred, zero_division=0), 4),
                 recall    = round(recall_score(y_test, y_pred, zero_division=0), 4),
                 f1        = round(f1_score(y_test, y_pred, zero_division=0), 4),
                 confusion_matrix = confusion_matrix(y_test, y_pred).tolist(),
                 model_obj = mdl)
        print(f"  {name:<23} {r['accuracy']:>7.4f} {r['precision']:>7.4f}"
              f" {r['recall']:>8.4f} {r['f1']:>7.4f}")
        results.append(r)
    print(f"{'─'*62}")
    print("  Recall is most important — minimises missed heart disease cases")
    return results


def save_best(results, scaler, feature_cols):
    print("\n[5/5] Saving best model ...")
    best = max(results, key=lambda r: r["recall"])
    print(f"  Best → {best['name']}  (Recall={best['recall']:.4f})")
    cm = best["confusion_matrix"]
    print(f"  Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} | FN={cm[1][0]} TP={cm[1][1]}")

    joblib.dump(best["model_obj"], os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler,            os.path.join(MODEL_DIR, "scaler.pkl"))

    meta = {k: v for k, v in best.items() if k != "model_obj"}
    meta["all_results"] = [{k: v for k, v in r.items() if k != "model_obj"} for r in results]
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)

    print("  Saved: best_model.pkl  scaler.pkl  model_meta.json  feature_cols.json")


if __name__ == "__main__":
    print("="*62)
    print("  Heart Disease Risk Predictor — Training Pipeline")
    print("="*62)
    df = load_and_clean(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)
    trained = train_all(X_train, y_train)
    results = evaluate_all(trained, X_test, y_test)
    save_best(results, scaler, feature_cols)
    print("\n" + "="*62)
    print("  Training complete! Now run:  streamlit run app.py")
    print("="*62 + "\n")

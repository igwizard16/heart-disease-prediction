"""
preprocessing.py
Handles all data loading, cleaning, scaling, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# ─────────────────────────────────────────────
# Column metadata  (human labels, value maps)
# ─────────────────────────────────────────────

FEATURE_META = {
    "HighBP":           {"label": "High Blood Pressure",          "type": "binary"},
    "HighChol":         {"label": "High Cholesterol",             "type": "binary"},
    "CholCheck":        {"label": "Cholesterol Check in 5 Years", "type": "binary"},
    "BMI":              {"label": "Body Mass Index (BMI)",        "type": "numerical"},
    "Smoker":           {"label": "Smoked ≥100 Cigarettes Ever",  "type": "binary"},
    "Stroke":           {"label": "Ever Had a Stroke",            "type": "binary"},
    "Diabetes":         {"label": "Diabetes Status",              "type": "categorical",
                         "values": {0: "No Diabetes", 1: "Pre-Diabetic", 2: "Diabetic"}},
    "PhysActivity":     {"label": "Physical Activity (last 30 days)", "type": "binary"},
    "Fruits":           {"label": "Eats Fruits Daily",            "type": "binary"},
    "Veggies":          {"label": "Eats Vegetables Daily",        "type": "binary"},
    "HvyAlcoholConsump":{"label": "Heavy Alcohol Consumption",    "type": "binary"},
    "AnyHealthcare":    {"label": "Has Any Healthcare Coverage",  "type": "binary"},
    "NoDocbcCost":      {"label": "Skipped Doctor Due to Cost",   "type": "binary"},
    "GenHlth":          {"label": "General Health (1=Excellent, 5=Poor)", "type": "ordinal"},
    "MentHlth":         {"label": "Poor Mental Health Days (last 30)", "type": "numerical"},
    "PhysHlth":         {"label": "Poor Physical Health Days (last 30)", "type": "numerical"},
    "DiffWalk":         {"label": "Difficulty Walking/Climbing Stairs", "type": "binary"},
    "Sex":              {"label": "Sex",                          "type": "binary",
                         "values": {0: "Female", 1: "Male"}},
    "Age":              {"label": "Age Group",                   "type": "ordinal",
                         "values": {
                             1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39",
                             5: "40–44", 6: "45–49", 7: "50–54", 8: "55–59",
                             9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80+"}},
    "Education":        {"label": "Education Level",             "type": "ordinal",
                         "values": {
                             1: "Never Attended School", 2: "Grades 1–8",
                             3: "Grades 9–11", 4: "Grade 12 / GED",
                             5: "Some College", 6: "College Graduate"}},
    "Income":           {"label": "Household Income",            "type": "ordinal",
                         "values": {
                             1: "< $10,000", 2: "$10,000–$14,999", 3: "$15,000–$19,999",
                             4: "$20,000–$24,999", 5: "$25,000–$34,999",
                             6: "$35,000–$49,999", 7: "$50,000–$74,999", 8: "≥ $75,000"}},
}

# Features that need StandardScaler
NUMERICAL_FEATURES = ["BMI", "MentHlth", "PhysHlth"]

TARGET_COL = "HeartDiseaseorAttack"


# ─────────────────────────────────────────────
# Main functions
# ─────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV. Raises FileNotFoundError if path is wrong."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop duplicates
    - Drop rows with missing values (there are none in BRFSS2015 but defensive)
    - Cast target to int
    """
    df = df.drop_duplicates()
    df = df.dropna()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df.reset_index(drop=True)


def split_and_scale(df: pd.DataFrame, test_size=0.2, random_state=42, use_smote=True):
    """
    Returns:
        X_train, X_test, y_train, y_test, scaler
    
    Steps:
      1. Split features / target
      2. Train/test split (stratified to preserve imbalance ratio)
      3. StandardScaler on numerical cols
      4. SMOTE on training set only (NEVER on test set)
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values.astype(float)
    y = df[TARGET_COL].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale only numerical columns (by index in feature_cols list)
    num_idx = [feature_cols.index(f) for f in NUMERICAL_FEATURES if f in feature_cols]
    scaler = StandardScaler()
    X_train[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
    X_test[:, num_idx]  = scaler.transform(X_test[:, num_idx])   # use same scaler

    if use_smote:
        sm = SMOTE(random_state=random_state, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def save_scaler(scaler, path: str):
    joblib.dump(scaler, path)


def load_scaler(path: str):
    return joblib.load(path)


def preprocess_user_input(user_dict: dict, feature_cols: list, scaler) -> np.ndarray:
    """
    Convert a {feature: value} dict from the UI into a scaled numpy array
    ready for model.predict().
    """
    row = np.array([float(user_dict[f]) for f in feature_cols]).reshape(1, -1)
    num_idx = [feature_cols.index(f) for f in NUMERICAL_FEATURES if f in feature_cols]
    row[:, num_idx] = scaler.transform(row[:, num_idx])
    return row

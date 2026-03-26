# Heart Disease Risk Predictor
**BRFSS 2015 Dataset | Streamlit Web App | Academic ML Project**

---

## What This Project Does

This app predicts heart disease risk based on 21 health indicators from the CDC BRFSS 2015 survey.  
It uses machine learning (Logistic Regression, Random Forest, XGBoost) and displays:

- Risk level: 🟢 Low / 🟡 Medium / 🔴 High
- Risk factors you should be aware of
- Protective factors you have
- General wellness suggestions

---

## Project Structure

```
heart_disease_project/
│
├── app.py                  ← Streamlit web app (the UI)
├── requirements.txt        ← All Python packages needed
├── README.md               ← This file
│
├── src/
│   ├── preprocessing.py    ← Data loading, cleaning, scaling, SMOTE
│   └── train_model.py      ← Train all models, save best one
│
├── models/                 ← Created automatically after training
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_cols.json
│   └── model_meta.json
│
└── data/
    └── heart_disease_health_indicators_BRFSS2015.csv   ← Put CSV here
```

---

## Step-by-Step Setup (Do This Exactly)

### Step 1 — Make sure Python is installed
Open your terminal and type:
```
python --version
```
You need Python 3.9 or higher. If not installed, download from https://python.org

---

### Step 2 — Go to the project folder
```bash
cd heart_disease_project
```
(Replace with the actual path where you saved this folder)

---

### Step 3 — Install all packages
```bash
pip install -r requirements.txt
```
This installs: streamlit, scikit-learn, xgboost, imbalanced-learn, pandas, numpy, joblib

If you get a permission error, use:
```bash
pip install -r requirements.txt --user
```

---

### Step 4 — Put the dataset in the right place
Make sure this file exists:
```
data/heart_disease_health_indicators_BRFSS2015.csv
```
The `data/` folder must be inside `heart_disease_project/`.

---

### Step 5 — Train the model (run ONCE)
```bash
python src/train_model.py
```

This will take **3–5 minutes** (SMOTE on 200k rows takes time).  
You will see output like:

```
[1/5] Loading data ...
[2/5] Splitting + Scaling + SMOTE ...
[3/5] Training 3 models ...
      Logistic Regression ... done
      Random Forest ... done
      XGBoost ... done
[4/5] Evaluation Report
[5/5] Saving best model ...
Training complete! Now run:  streamlit run app.py
```

After this, a `models/` folder will be created with 4 files.  
**You only need to do this once.** After that, the app loads the saved model.

---

### Step 6 — Launch the web app
```bash
streamlit run app.py
```

Your browser will automatically open at:  
**http://localhost:8501**

If it doesn't open, manually go to that URL.

---

## How to Use the App

1. Fill in all the fields in the form (the sidebar has a health reference guide)
2. Click **"Predict My Risk"** at the bottom
3. See your risk level, risk factors, and wellness suggestions

---

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** ✅ | 74.3% | 25.5% | **77.4%** | 38.4% |
| Random Forest | 83.1% | 31.8% | 55.7% | 40.5% |
| XGBoost | 89.6% | 49.2% | 15.3% | 23.3% |

**Why Logistic Regression?**  
In medical screening, **Recall (sensitivity)** is the most important metric.  
A missed heart disease case (False Negative) is far more dangerous than a false alarm.  
LR catches 77% of actual heart disease cases vs XGBoost's 15%.

---

## Key Fixes Over Original Notebook

| Problem in Original | Fix Applied |
|---|---|
| SMOTE applied to entire dataset | SMOTE applied ONLY to training data |
| Scaler fit on full data (data leakage) | Scaler fit only on X_train, then transform X_test |
| Model selected by accuracy | Model selected by Recall |
| Raw variable names in UI (`HighBP`, `GenHlth`) | Human-readable labels with tooltips |
| No health context for user | Sidebar with BMI, BP, Cholesterol reference guide |
| Binary Yes/No prediction | Risk probability + Low/Medium/High + explanations |
| No input validation | Input validated, app won't crash on edge cases |

---

## Common Errors & Fixes

**Error:** `FileNotFoundError: Dataset not found`  
**Fix:** Make sure the CSV is inside the `data/` folder, not anywhere else.

**Error:** `ModuleNotFoundError: No module named 'streamlit'`  
**Fix:** Run `pip install -r requirements.txt` again.

**Error:** `FileNotFoundError: best_model.pkl not found`  
**Fix:** You forgot to run `python src/train_model.py` first.

**Error:** Port 8501 already in use  
**Fix:** `streamlit run app.py --server.port 8502`

---

## Disclaimer

This is an academic project for educational purposes only.  
This tool does NOT provide medical advice, diagnosis, or treatment.  
Always consult a qualified healthcare professional for medical decisions.

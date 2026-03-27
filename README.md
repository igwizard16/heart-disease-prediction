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

Link to open the website - https://heart-disease-prediction-aabsqsbiknuhmwczzn8qwn.streamlit.app/

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

## Disclaimer

This is an academic project for educational purposes only.  
This tool does NOT provide medical advice, diagnosis, or treatment.  
Always consult a qualified healthcare professional for medical decisions.

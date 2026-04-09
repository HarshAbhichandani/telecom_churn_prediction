"""
predict.py — Load the saved model and predict churn for new customers
"""

import pandas as pd
import numpy as np
import joblib


MODELS_DIR = "../models"


def load_artifacts():
    model        = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    scaler       = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    feature_names = joblib.load(f"{MODELS_DIR}/feature_names.pkl")
    return model, scaler, feature_names


def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn probability for a single customer.

    Parameters
    ----------
    customer_data : dict  — raw feature values (same format as training data)

    Returns
    -------
    dict with 'churn_prediction' (0/1) and 'churn_probability' (float)
    """
    model, scaler, feature_names = load_artifacts()
    df = pd.DataFrame([customer_data])

    # Align columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    X_scaled = scaler.transform(df)
    prediction   = model.predict(X_scaled)[0]
    probability  = model.predict_proba(X_scaled)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": "HIGH" if probability > 0.6 else "MEDIUM" if probability > 0.3 else "LOW",
    }


if __name__ == "__main__":
    # Example customer
    sample = {
        "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
        "tenure": 5, "PhoneService": 1, "MultipleLines": 0,
        "InternetService": 2, "OnlineSecurity": 0, "OnlineBackup": 0,
        "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 1,
        "StreamingMovies": 1, "Contract": 0, "PaperlessBilling": 1,
        "PaymentMethod": 0, "MonthlyCharges": 85.5, "TotalCharges": 430.0,
        "AvgMonthlySpend": 86.0, "TenureGroup": 0,
        "NumServices": 4, "IsHighValue": 1, "HasProtection": 0,
    }
    result = predict_churn(sample)
    print(f"Churn Prediction : {result['churn_prediction']} ({'Will Churn' if result['churn_prediction'] else 'Will Stay'})")
    print(f"Churn Probability: {result['churn_probability']:.1%}")
    print(f"Risk Level       : {result['risk_level']}")

"""
preprocess.py — Data cleaning and feature engineering for Telco Churn dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Telco churn dataset."""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset:
      - Drop customerID
      - Fix TotalCharges (spaces → NaN → numeric)
      - Impute missing TotalCharges with median
      - Encode Churn as binary (1 = Yes, 0 = No)
    """
    df = df.copy()

    # Drop ID column
    df.drop("customerID", axis=1, inplace=True, errors="ignore")

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_count = df["TotalCharges"].isnull().sum()
    if missing_count > 0:
        median_val = df["TotalCharges"].median()
        df["TotalCharges"].fillna(median_val, inplace=True)
        print(f"✅ Imputed {missing_count} missing TotalCharges with median ({median_val:.2f})")

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    print(f"✅ Churn encoded | Churn rate: {df['Churn'].mean():.1%}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 5 new features from existing data:
      1. AvgMonthlySpend — TotalCharges / tenure
      2. TenureGroup     — binned tenure (0–3)
      3. NumServices     — count of active services
      4. IsHighValue     — MonthlyCharges > median
      5. HasProtection   — any protection/support service active
    """
    df = df.copy()

    df["AvgMonthlySpend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    ).astype(float)

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)

    df["IsHighValue"] = (
        df["MonthlyCharges"] > df["MonthlyCharges"].median()
    ).astype(int)

    protection_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["HasProtection"] = (df[protection_cols] == "Yes").any(axis=1).astype(int)

    print(f"✅ Feature engineering done | Shape: {df.shape}")
    return df


def encode_and_scale(df: pd.DataFrame, fit_scaler: bool = True):
    """
    Label-encode all object columns, then scale numeric features.

    Returns
    -------
    X : pd.DataFrame  — encoded + scaled features
    y : pd.Series     — target (Churn)
    scaler : StandardScaler
    """
    df = df.copy().dropna()

    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X) if fit_scaler else scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )

    return X_scaled, y, scaler

"""
train.py — Train and compare ML models for churn prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
import joblib
import os

from preprocess import load_data, clean_data, engineer_features, encode_and_scale


DATA_PATH  = "../data/telco_churn.csv"
MODELS_DIR = "../models"


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_split=20),
        "Random Forest":       RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, n_jobs=-1),
    }


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred),    4),
        "F1-Score":  round(f1_score(y_true, y_pred),        4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob),   4),
    }


def train_and_evaluate():
    # ── Pipeline ──────────────────────────────────────────────────────────────
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    X, y, scaler = encode_and_scale(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}\n")

    os.makedirs(MODELS_DIR, exist_ok=True)
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"🤖 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_prob)
        metrics["Model"] = name
        results.append(metrics)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        print(f"   CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   Test  | Acc: {metrics['Accuracy']} | F1: {metrics['F1-Score']} | AUC: {metrics['ROC-AUC']}\n")

    # ── Results table ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).set_index("Model")
    results_df.to_csv(f"{MODELS_DIR}/model_results.csv")
    print("📊 MODEL COMPARISON:\n", results_df.to_string(), "\n")

    # ── Save best model ────────────────────────────────────────────────────────
    best_name = results_df["ROC-AUC"].idxmax()
    best_model = models[best_name]
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(scaler,     f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(list(X.columns), f"{MODELS_DIR}/feature_names.pkl")
    print(f"🏆 Best model: {best_name} (AUC={results_df.loc[best_name,'ROC-AUC']})")
    print(f"✅ Saved to {MODELS_DIR}/")

    return results_df


if __name__ == "__main__":
    train_and_evaluate()

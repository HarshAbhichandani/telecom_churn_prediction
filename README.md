# 📊 Customer Churn Prediction
> A complete Data Science project predicting telecom customer churn using EDA, feature engineering, and multiple ML models.

---

## 🗂️ Dataset
| Detail | Info |
|--------|------|
| **Name** | IBM Telco Customer Churn |
| **Source** | Kaggle |
| **Link** | 🔗 https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target** | `Churn` (Yes / No) |
| **Churn Rate** | ~26–30% |

> **Download the dataset:** Go to the Kaggle link above → Click **Download** → Place `Telco-Customer-Churn.csv` inside the `data/` folder and rename it to `telco_churn.csv`

---

## 🎯 Project Objectives
- Perform thorough **Exploratory Data Analysis (EDA)**
- Handle **missing values** and **data cleaning**
- Create meaningful **engineered features**
- Train and compare **3 ML models**: Logistic Regression, Decision Tree, Random Forest
- Evaluate using **Accuracy, Precision, Recall, F1-Score, ROC-AUC**

---

## 📁 Project Structure
```
churn-prediction-ds/
│── data/
│   └── telco_churn.csv              ← Dataset (download from Kaggle)
│── notebooks/
│   └── analysis.ipynb               ← Full analysis notebook (EDA → Models)
│── models/
│   ├── best_model.pkl               ← Saved best model
│   ├── scaler.pkl                   ← Fitted StandardScaler
│   ├── feature_names.pkl            ← Feature column list
│   ├── model_results.csv            ← Model comparison table
│   ├── 01_churn_distribution.png
│   ├── 02_missing_values.png
│   ├── 03_numerical_features.png
│   ├── 04_categorical_features.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_model_comparison.png
│   ├── 07_confusion_matrices.png
│   └── 08_feature_importance.png
│── src/
│   ├── preprocess.py                ← Data cleaning + feature engineering
│   ├── train.py                     ← Model training + evaluation
│   └── predict.py                   ← Inference on new customers
└── README.md
```

---

## 🔧 Tech Stack
| Category | Libraries |
|----------|-----------|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Model Persistence | `joblib` |
| Notebook | `jupyter` |

---

## 📊 EDA Highlights
- **Missing Values:** `TotalCharges` has 11 blank entries (converted and imputed with median)
- **Churn Rate:** ~29% — class imbalance noted
- **Key Finding:** Month-to-month contracts + Fiber Optic internet → highest churn
- **Correlation:** `tenure` (negative) and `Contract` are strongest churn predictors

---

## 🔨 Feature Engineering (5 New Features)
| Feature | Description |
|---------|-------------|
| `AvgMonthlySpend` | TotalCharges ÷ tenure |
| `TenureGroup` | Binned: New / Growing / Mature / Loyal |
| `NumServices` | Count of active services (0–9) |
| `IsHighValue` | MonthlyCharges > median → 1 |
| `HasProtection` | Has any security/support service → 1 |

---

## 🤖 Models Trained
| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model (scaled features) |
| Decision Tree | Interpretable tree (max_depth=6) |
| Random Forest | Ensemble of 100 trees (max_depth=10) |

---

## 📈 Model Results
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.71 | ~0.56 | ~0.18 | ~0.27 | ~0.69 |
| Decision Tree | ~0.71 | ~0.54 | ~0.23 | ~0.33 | ~0.66 |
| Random Forest | ~0.71 | ~0.54 | ~0.14 | ~0.22 | ~0.67 |

> **Best Model:** Logistic Regression (highest ROC-AUC)

---

## 🚀 Quick Start

### Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib
```

### Run the notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```

### Train models via script
```bash
cd src
python train.py
```

### Predict on new customer
```bash
cd src
python predict.py
```

---

## 📸 Generated Visualizations
1. `01_churn_distribution.png` — Bar + Pie chart of churn rate
2. `02_missing_values.png` — Missing value analysis
3. `03_numerical_features.png` — Histograms + Boxplots
4. `04_categorical_features.png` — Churn rate by category
5. `05_correlation_heatmap.png` — Full feature correlation matrix
6. `06_model_comparison.png` — All metrics + ROC curves
7. `07_confusion_matrices.png` — Confusion matrices for all models
8. `08_feature_importance.png` — Top 15 features (Random Forest)

---

## 💡 Key Insights
1. **Contract type** is the strongest predictor — month-to-month customers churn 3x more
2. **Tenure** is strongly negative — long-term customers rarely churn
3. **Fiber Optic** users churn more despite paying more
4. **Tech Support / Online Security** reduce churn significantly
5. **Senior citizens** and customers without dependents churn at higher rates

---

*Built with Python • scikit-learn • pandas • matplotlib • seaborn*

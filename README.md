# 📊 Business Bankruptcy Risk Prediction (ML)

A complete machine learning pipeline that predicts company bankruptcy risk using financial ratios. Built with **Python + R**, featuring 5 ML models, statistical analysis, and an interactive prediction web app.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![R](https://img.shields.io/badge/R-4.0+-green.svg)](https://r-project.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project builds and compares multiple machine learning models for predicting business bankruptcy, following the methodology from:

- **Olson & Wu (2017)** — *Enterprise Risk Management Models* (Decision Tree, Logistic Regression, Neural Network)
- **Sundararajan (2025)** — *Multivariate Analysis and Machine Learning Techniques* (Ensemble methods, Feature Selection)

### Dataset

**Taiwan Economic Journal Company Bankruptcy Prediction** ([Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction))

| Property | Value |
|----------|-------|
| Records | 6,819 companies |
| Features | 95 financial ratios |
| Target | Binary (Bankrupt: 3.2%, Survived: 96.8%) |
| Period | 1999–2009 |

### Results

| Model | Accuracy | Recall | F1-Score | AUC-ROC |
|-------|----------|--------|----------|---------|
| Logistic Regression | 0.872 | 0.773 | 0.281 | 0.917 |
| Decision Tree | 0.890 | 0.818 | 0.324 | 0.906 |
| **Random Forest** ★ | **0.932** | **0.773** | **0.422** | **0.945** |
| XGBoost | 0.930 | 0.682 | 0.387 | 0.918 |
| Neural Network | 0.960 | 0.432 | 0.409 | 0.881 |

★ **Best model: Random Forest** (highest F1-Score + AUC-ROC + cross-validation stability)

## Project Structure

```
business-risk-ml/
├── data/
│   ├── data.csv                    # Raw Kaggle dataset (download separately)
│   ├── README.md                   # Dataset download instructions
│   └── processed/                  # Preprocessed train/test splits
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── metadata.json
├── notebooks/
│   ├── 01_eda.py                   # Step 2: Exploratory Data Analysis
│   ├── 02_preprocessing.py         # Step 3: Data Preprocessing Pipeline
│   ├── 03_model_training.py        # Step 4: Train 5 ML Models
│   └── 04_evaluation.py            # Step 5: Model Evaluation & Comparison
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data loading & validation utilities
│   └── preprocessor.py             # Reusable preprocessing functions
├── R_analysis/
│   ├── install_packages.R          # R package installation
│   ├── load_data.R                 # Load Python-processed data into R
│   └── model_comparison.R          # R statistical analysis (Step 5)
├── models/
│   ├── best_model.pkl              # Best performing model (Random Forest)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── neural_network.pkl
│   └── scaler.pkl                  # StandardScaler for new predictions
├── app/
│   └── streamlit_app.py            # Step 6: Interactive prediction web app
├── figures/                        # All generated charts (01-15)
├── config.py                       # Project configuration
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/OVuyo/business-risk-ml.git
cd business-risk-ml

# Python
pip install -r requirements.txt

# R (optional, for statistical analysis)
Rscript R_analysis/install_packages.R
```

### 2. Download Data

Download `data.csv` from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction) and place it in `data/`.

### 3. Run the Pipeline

```bash
# Step 2: Exploratory Data Analysis
python notebooks/01_eda.py

# Step 3: Preprocessing (outliers, feature selection, SMOTE, scaling)
python notebooks/02_preprocessing.py

# Step 4: Train all 5 models
python notebooks/03_model_training.py

# Step 5: Evaluate and compare models
python notebooks/04_evaluation.py

# Step 5b: R statistical analysis (optional)
Rscript R_analysis/model_comparison.R

# Step 6: Launch the prediction web app
streamlit run app/streamlit_app.py
```

## Pipeline Details

### Preprocessing (Step 3)

```
Raw Data (6,819 × 96)
  → Winsorization (cap outliers at 1st/99th percentile)
  → Multicollinearity removal (drop features with |r| > 0.95)
  → Feature selection (top 30 by Mutual Information)
  → Stratified train/test split (80/20)
  → SMOTE oversampling (176 → 5,279 bankrupt examples)
  → StandardScaler normalization
Clean Data (10,558 × 30 training / 1,364 × 30 testing)
```

### Models (Step 4)

1. **Logistic Regression** — Linear baseline with L2 regularization
2. **Decision Tree** — Interpretable rules (max_depth=8, like Rattle/R)
3. **Random Forest** — 200 trees with √features per split (bagging)
4. **XGBoost** — Gradient boosted trees (sequential boosting)
5. **Neural Network** — MLP with 3 hidden layers (64→32→16, ReLU)

### Key Findings

- **Top predictors**: Persistent EPS, Net Income ratios, Borrowing dependency
- **Class imbalance** (30:1) addressed with SMOTE — critical for recall
- **Random Forest** achieves best balance of recall (77.3%) and precision (29.1%)
- **Decision Tree** has highest recall (81.8%) but lowest precision
- Results align with Olson & Wu's finding that ensemble methods outperform single models

## Technologies

**Python**: pandas, scikit-learn, XGBoost, Keras/TensorFlow, matplotlib, seaborn, Streamlit, imbalanced-learn

**R**: caret, randomForest, rpart, ggplot2, corrplot, pROC

## References

1. Olson, D.L. & Wu, D.D. (2017). *Enterprise Risk Management Models*. Springer.
2. Sundararajan, S. (2025). *Multivariate Analysis and Machine Learning Techniques*. Springer.
3. Olson, D.L., Delen, D. & Meng, Y. (2012). Comparative analysis of data mining methods for bankruptcy prediction. *Decision Support Systems*, 52(2), 464–473.
4. Dataset: [Taiwan Economic Journal](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction) via Kaggle.

## License

MIT License — see [LICENSE](LICENSE) for details.

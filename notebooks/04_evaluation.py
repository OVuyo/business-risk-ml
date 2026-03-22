"""
=============================================================
 STEP 5: MODEL EVALUATION & COMPARISON
 Business Risk ML Project
=============================================================
 Deep evaluation of all 5 models:
   - Classification reports
   - ROC & Precision-Recall analysis
   - Feature importance comparison
   - Cross-validation stability check
   - Final model recommendation

 Olson & Wu (Ch.9): "Data mining practice is usually to run all
 three models and compare results... different models will yield
 different results, and these relative advantages are liable to
 change with new data."
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

# ── Load Data & Models ───────────────────────────────────────
print("=" * 60)
print("  STEP 5: MODEL EVALUATION")
print("=" * 60)

X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').iloc[:, 0]
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').iloc[:, 0]

model_files = {
    'Logistic Regression': 'models/logistic_regression.pkl',
    'Decision Tree': 'models/decision_tree.pkl',
    'Random Forest': 'models/random_forest.pkl',
    'XGBoost': 'models/xgboost_model.pkl',
    'Neural Network': 'models/neural_network.pkl',
}

models = {name: joblib.load(path) for name, path in model_files.items()}
features = X_train.columns.tolist()


# ── 5.1 Full Classification Reports ──────────────────────────
print("\n── 5.1 Classification Reports ──\n")

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n  {name}:")
    print("  " + "-" * 50)
    report = classification_report(y_test, y_pred, target_names=['Survived', 'Bankrupt'])
    for line in report.split('\n'):
        print(f"  {line}")


# ── 5.2 Cross-Validation (5-Fold) ────────────────────────────
print("\n── 5.2 Cross-Validation Stability ──")
print("  (5-fold CV on training data to check for overfitting)\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:25s}  F1: {scores.mean():.4f} ± {scores.std():.4f}  "
          f"[{scores.min():.4f} — {scores.max():.4f}]")


# ── 5.3 Comparison Table ─────────────────────────────────────
print("\n── 5.3 Final Comparison Table ──\n")

results = pd.read_csv('data/processed/model_results.csv')
comparison = results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].copy()
comparison['CV F1 (mean)'] = [cv_results[m].mean() for m in comparison['Model']]
comparison['CV F1 (std)'] = [cv_results[m].std() for m in comparison['Model']]

# Format for display
print(comparison.to_string(index=False, float_format='%.4f'))

# Olson & Wu style comparison table
print(f"\n  Olson & Wu Style Comparison (like Table 9.8):")
print(f"  {'Model':25s} {'Correct Survived':>18s} {'Correct Bankrupt':>18s} {'Overall':>10s}")
print(f"  {'─'*75}")
for _, row in results.iterrows():
    tn, fp, fn, tp = row['TN'], row['FP'], row['FN'], row['TP']
    surv_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    bank_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    overall = (tn + tp) / (tn + fp + fn + tp)
    print(f"  {row['Model']:25s} {surv_acc:18.3f} {bank_acc:18.3f} {overall:10.3f}")


# ── 5.4 Final Recommendation ─────────────────────────────────
print(f"\n── 5.4 Model Recommendation ──")

# Score models on multiple criteria
score_board = pd.DataFrame()
score_board['Model'] = comparison['Model']
score_board['F1'] = comparison['F1-Score'].rank(ascending=True)
score_board['AUC'] = comparison['AUC-ROC'].rank(ascending=True)
score_board['Recall'] = comparison['Recall'].rank(ascending=True)  
score_board['CV_Stability'] = (1 / comparison['CV F1 (std)']).rank(ascending=True)
score_board['Total'] = score_board[['F1', 'AUC', 'Recall', 'CV_Stability']].sum(axis=1)

best = score_board.loc[score_board['Total'].idxmax(), 'Model']

print(f"\n  ★ RECOMMENDED MODEL: {best}")
print(f"  Scoring: F1 + AUC + Recall + CV Stability")
print(f"\n  Rationale:")
print(f"  - For bankruptcy detection, RECALL is critical (catching actual bankruptcies)")
print(f"  - AUC-ROC measures discrimination across all thresholds")
print(f"  - F1 balances precision and recall")
print(f"  - CV stability ensures the model generalizes well")

# Save comparison
comparison.to_csv('data/processed/final_comparison.csv', index=False)

print(f"\n" + "=" * 60)
print(f"  STEP 5 COMPLETE")
print(f"=" * 60)

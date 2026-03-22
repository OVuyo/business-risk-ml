"""
=============================================================
 STEP 4: MODEL TRAINING
 Business Risk ML Project
=============================================================
 Train 5 classification models on the preprocessed data:
   1. Logistic Regression  (linear baseline)
   2. Decision Tree         (interpretable, non-linear)
   3. Random Forest         (ensemble bagging)
   4. XGBoost               (ensemble boosting)
   5. Neural Network        (deep learning)

 Following Olson & Wu (ERM Ch.9): "Data mining for classification
 models have three basic tools — decision trees, logistic regression,
 and neural network models."
 
 We add Random Forest and XGBoost because Sundararajan (Ch.11):
 "Ensemble methods involve using a set of multiple classifiers...
 combining their predictions with improved accuracy and robustness."
=============================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Neural network
from sklearn.neural_network import MLPClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Load Preprocessed Data ───────────────────────────────────
print("=" * 60)
print("  STEP 4: MODEL TRAINING")
print("=" * 60)

X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').iloc[:, 0]
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').iloc[:, 0]

print(f"\n  Training: {X_train.shape[0]:,} × {X_train.shape[1]} features")
print(f"  Testing:  {X_test.shape[0]:,} × {X_test.shape[1]} features")
print(f"  Train balance: {(y_train==0).sum():,} survived / {(y_train==1).sum():,} bankrupt")
print(f"  Test balance:  {(y_test==0).sum():,} survived / {(y_test==1).sum():,} bankrupt")

features = X_train.columns.tolist()
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)


# ════════════════════════════════════════════════════════════
# MODEL 1: LOGISTIC REGRESSION
# ════════════════════════════════════════════════════════════
"""
WHAT: Linear model that predicts probability of bankruptcy using
      a weighted sum of features passed through a sigmoid function.
      P(bankrupt) = sigmoid(w1*x1 + w2*x2 + ... + b)

WHY:  The simplest classification model — our BASELINE.
      If other models can't beat this, they're not worth the complexity.
      Olson & Wu used this as one of their three core models.

KEY PARAMS:
      C=1.0        → regularization strength (prevents overfitting)
      max_iter=1000 → enough iterations for convergence
      penalty='l2'  → Ridge regularization (shrinks weights, keeps all features)
"""
print("\n── Model 1: Logistic Regression ──")
t0 = time.time()

lr_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    random_state=RANDOM_STATE
)
lr_model.fit(X_train, y_train)
lr_time = time.time() - t0

joblib.dump(lr_model, 'models/logistic_regression.pkl')
print(f"  ✓ Trained in {lr_time:.2f}s")


# ════════════════════════════════════════════════════════════
# MODEL 2: DECISION TREE
# ════════════════════════════════════════════════════════════
"""
WHAT: Splits data into branches based on feature thresholds.
      "If revenue < 78, predict NOT bankrupt" (from Olson & Wu Ch.9)

WHY:  Most INTERPRETABLE model — you can literally read the rules.
      This is what Rattle produced in the textbook's demonstration.

KEY PARAMS:
      max_depth=8     → prevents the tree from memorizing training data
      min_samples_leaf=20 → each leaf must have ≥20 samples (Olson & Wu default)
      class_weight='balanced' → extra penalty for misclassifying bankrupt cases
"""
print("\n── Model 2: Decision Tree ──")
t0 = time.time()

dt_model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    min_samples_split=40,
    class_weight='balanced',
    random_state=RANDOM_STATE
)
dt_model.fit(X_train, y_train)
dt_time = time.time() - t0

joblib.dump(dt_model, 'models/decision_tree.pkl')
print(f"  ✓ Trained in {dt_time:.2f}s")

# Print tree rules (like Rattle's graphical tree)
tree_rules = export_text(dt_model, feature_names=features, max_depth=3)
print(f"  Tree rules (first 3 levels):\n{tree_rules[:500]}")


# ════════════════════════════════════════════════════════════
# MODEL 3: RANDOM FOREST
# ════════════════════════════════════════════════════════════
"""
WHAT: Ensemble of 200 decision trees, each trained on a random
      subset of data and features. Final prediction = majority vote.

WHY:  Sundararajan (Ch.11): "Compared to a single classifier, bagging
      improves accuracy and is less affected by noise."
      Random Forest is the gold standard for tabular classification.

KEY PARAMS:
      n_estimators=200     → number of trees (more = better, but slower)
      max_depth=12         → each tree goes deeper than the single tree
      min_samples_leaf=10  → leaves can be smaller since we have 200 trees
      max_features='sqrt'  → each tree sees only √30 ≈ 5 features per split
                              (this decorrelates the trees — the key RF insight)
"""
print("\n── Model 3: Random Forest ──")
t0 = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1  # Use all CPU cores
)
rf_model.fit(X_train, y_train)
rf_time = time.time() - t0

joblib.dump(rf_model, 'models/random_forest.pkl')
print(f"  ✓ Trained in {rf_time:.2f}s (200 trees)")


# ════════════════════════════════════════════════════════════
# MODEL 4: XGBOOST
# ════════════════════════════════════════════════════════════
"""
WHAT: Gradient Boosted Trees — builds trees SEQUENTIALLY, where
      each new tree focuses on the mistakes of previous trees.

WHY:  Sundararajan (Ch.11): "Boosting is an improvisation of bagging.
      The misclassified data is given added weightage after each model."
      XGBoost consistently wins ML competitions on tabular data.

KEY PARAMS:
      n_estimators=200      → number of boosting rounds
      max_depth=6           → shallower trees (boosting works with weak learners)
      learning_rate=0.1     → how much each tree contributes (lower = more conservative)
      scale_pos_weight=30   → compensates for 30:1 imbalance in original data
      subsample=0.8         → each tree sees 80% of data (prevents overfitting)
      colsample_bytree=0.8  → each tree sees 80% of features
"""
print("\n── Model 4: XGBoost ──")
t0 = time.time()

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - t0

joblib.dump(xgb_model, 'models/xgboost_model.pkl')
print(f"  ✓ Trained in {xgb_time:.2f}s")


# ════════════════════════════════════════════════════════════
# MODEL 5: NEURAL NETWORK (MLP)
# ════════════════════════════════════════════════════════════
"""
WHAT: Multi-Layer Perceptron — a feed-forward neural network with
      3 hidden layers: 64 → 32 → 16 neurons.

WHY:  Olson & Wu and Sundararajan both include neural networks
      as their third core model. NNs capture complex non-linear
      interactions between features that simpler models miss.

KEY PARAMS:
      hidden_layer_sizes=(64, 32, 16) → 3 hidden layers
      activation='relu'    → Rectified Linear Unit (standard for modern NNs)
      solver='adam'         → Adam optimizer (adaptive learning rate)
      learning_rate_init=0.001 → initial step size
      max_iter=200          → training epochs
      early_stopping=True   → stop if validation score stops improving
      
      From Sundararajan (Ch.12): "Add a fully connected layer with a ReLU
      activation function... Add a sigmoid in the last layer for binary
      classification."
"""
print("\n── Model 5: Neural Network (MLP) ──")
t0 = time.time()

nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=RANDOM_STATE
)
nn_model.fit(X_train, y_train)
nn_time = time.time() - t0

joblib.dump(nn_model, 'models/neural_network.pkl')
print(f"  ✓ Trained in {nn_time:.2f}s ({nn_model.n_iter_} epochs)")


# ════════════════════════════════════════════════════════════
# EVALUATE ALL MODELS
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  MODEL EVALUATION ON TEST SET")
print("=" * 60)

models = {
    'Logistic Regression': (lr_model, lr_time),
    'Decision Tree': (dt_model, dt_time),
    'Random Forest': (rf_model, rf_time),
    'XGBoost': (xgb_model, xgb_time),
    'Neural Network': (nn_model, nn_time),
}

results = []

for name, (model, train_time) in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'Avg Precision': ap,
        'Train Time (s)': train_time,
        'TN': cm[0][0], 'FP': cm[0][1],
        'FN': cm[1][0], 'TP': cm[1][1],
    })
    
    print(f"\n  {name}:")
    print(f"    Accuracy:  {acc:.4f}    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}    F1-Score:  {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}    Avg Prec:  {ap:.4f}")
    print(f"    Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

results_df = pd.DataFrame(results)
results_df.to_csv('data/processed/model_results.csv', index=False)


# ── Find Best Model ──────────────────────────────────────────
best_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_f1 = results_df.loc[best_idx, 'F1-Score']
best_auc = results_df.loc[best_idx, 'AUC-ROC']

print(f"\n  ★ BEST MODEL: {best_model_name}")
print(f"    F1-Score: {best_f1:.4f} | AUC-ROC: {best_auc:.4f}")

# Save best model
best_model_obj = models[best_model_name][0]
joblib.dump(best_model_obj, 'models/best_model.pkl')
with open('models/best_model_name.txt', 'w') as f:
    f.write(best_model_name)


# ════════════════════════════════════════════════════════════
# VISUALIZATION: ROC Curves
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 1. ROC Curves
colors = ['#534AB7', '#1D9E75', '#378ADD', '#D85A30', '#D4537E']
for i, (name, (model, _)) in enumerate(models.items()):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f'{name} (AUC={auc_val:.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves — All Models', fontweight='bold')
axes[0].legend(fontsize=8, loc='lower right')
axes[0].grid(alpha=0.3)

# 2. Precision-Recall Curves
for i, (name, (model, _)) in enumerate(models.items()):
    y_prob = model.predict_proba(X_test)[:, 1]
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    axes[1].plot(rec_vals, prec_vals, color=colors[i], linewidth=2,
                label=f'{name} (AP={ap:.3f})')

axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves', fontweight='bold')
axes[1].legend(fontsize=8, loc='upper right')
axes[1].grid(alpha=0.3)

# 3. Model Comparison Bar Chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(metrics))
width = 0.15
for i, (_, row) in enumerate(results_df.iterrows()):
    vals = [row[m] for m in metrics]
    axes[2].bar(x + i * width, vals, width, label=row['Model'], color=colors[i], alpha=0.85)

axes[2].set_xticks(x + width * 2)
axes[2].set_xticklabels(metrics, fontsize=9)
axes[2].set_ylabel('Score')
axes[2].set_title('Model Comparison', fontweight='bold')
axes[2].legend(fontsize=7, loc='lower left')
axes[2].set_ylim(0, 1.05)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/09_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  ✓ Figure 9: Model comparison saved")


# ════════════════════════════════════════════════════════════
# VISUALIZATION: Confusion Matrices
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(22, 4))

for i, (name, (model, _)) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Survived', 'Bankrupt'],
                yticklabels=['Survived', 'Bankrupt'])
    axes[i].set_title(name, fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Actual' if i == 0 else '')
    axes[i].set_xlabel('Predicted')

plt.suptitle('Confusion Matrices — All Models (Test Set)', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('figures/10_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 10: Confusion matrices saved")


# ════════════════════════════════════════════════════════════
# VISUALIZATION: Feature Importance (Random Forest + XGBoost)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Random Forest importance
rf_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
rf_imp.tail(15).plot(kind='barh', ax=axes[0], color='#378ADD', edgecolor='white')
axes[0].set_title('Random Forest — Feature Importance', fontweight='bold')
axes[0].set_xlabel('Importance (Gini)')

# XGBoost importance
xgb_imp = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=True)
xgb_imp.tail(15).plot(kind='barh', ax=axes[1], color='#D85A30', edgecolor='white')
axes[1].set_title('XGBoost — Feature Importance', fontweight='bold')
axes[1].set_xlabel('Importance (Gain)')

plt.suptitle('Top 15 Features by Model Importance', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/11_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Figure 11: Feature importance saved")


# ── Final Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4 COMPLETE")
print("=" * 60)
print(f"\n  Models trained: 5")
print(f"  Best model:     {best_model_name} (F1={best_f1:.4f})")
print(f"  All models saved to: models/")
print(f"  Results saved to:    data/processed/model_results.csv")
print(f"  Figures saved to:    figures/09-11")
print(f"\n  → NEXT: Step 5 (R Statistical Analysis)")
print("=" * 60)

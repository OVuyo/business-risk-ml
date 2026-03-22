"""
=============================================================
 STEP 3: DATA PREPROCESSING & FEATURE ENGINEERING
 Business Risk ML Project
=============================================================
 This script takes the raw Kaggle data and prepares it for ML:
 
 PIPELINE:
   Raw Data (6,819 × 96)
     → 3.1  Outlier Treatment (Winsorization)
     → 3.2  Remove Redundant Features (multicollinearity)
     → 3.3  Feature Selection (top 30 by Mutual Information)
     → 3.4  Train/Test Split (80/20, stratified)
     → 3.5  SMOTE Oversampling (fix class imbalance)
     → 3.6  Feature Scaling (StandardScaler)
     → 3.7  Save Processed Data
   
   Clean Data ready for modeling!
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Reproducibility — same seed everywhere
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ════════════════════════════════════════════════════════════
# 3.0  LOAD RAW DATA
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 3: PREPROCESSING PIPELINE")
print("=" * 60)

df = pd.read_csv('data/data.csv')
df.columns = df.columns.str.strip()

TARGET = 'Bankrupt?'
features = [c for c in df.columns if c != TARGET]

print(f"\n  Input: {df.shape[0]:,} records × {df.shape[1]} columns")
print(f"  Target: '{TARGET}'")
print(f"  Bankrupt: {df[TARGET].sum()} ({df[TARGET].mean()*100:.1f}%)")


# ════════════════════════════════════════════════════════════
# 3.1  OUTLIER TREATMENT (Winsorization)
# ════════════════════════════════════════════════════════════
"""
WHAT: Cap extreme values at the 1st and 99th percentiles.

WHY:  13 features have max values in the BILLIONS (data entry errors).
      These would dominate distance-based models (KNN, SVM) and
      distort scaled features. Winsorization preserves the row while
      removing the extreme value's influence.

HOW:  For each feature, any value below Q1 becomes Q1, any value
      above Q99 becomes Q99. The middle 98% stays untouched.
      
EXAMPLE:
      Quick Ratio: max was 9,230,000,000 → becomes 0.066 (99th pct)
      The company still has the HIGHEST Quick Ratio, just not absurdly so.
"""
print("\n── 3.1 Outlier Treatment (Winsorization) ──")

outliers_fixed = 0
for col in features:
    q01 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    
    # Count how many values will be clipped
    n_clipped = ((df[col] < q01) | (df[col] > q99)).sum()
    outliers_fixed += n_clipped
    
    # Clip: values below q01 → q01, values above q99 → q99
    df[col] = df[col].clip(lower=q01, upper=q99)

print(f"  ✓ Winsorized {outliers_fixed:,} extreme values across {len(features)} features")
print(f"  ✓ All features now capped at [1st, 99th] percentile")


# ════════════════════════════════════════════════════════════
# 3.2  REMOVE REDUNDANT FEATURES (Multicollinearity)
# ════════════════════════════════════════════════════════════
"""
WHAT: Drop one feature from each pair where |correlation| > 0.95.

WHY:  Highly correlated features carry the SAME information.
      Including both:
      - Wastes model capacity
      - Makes feature importance unreliable
      - Can cause instability in Logistic Regression (collinearity)
      
HOW:  For each pair with |r| > 0.95, we keep the one that has
      HIGHER correlation with the target (more useful for prediction).
      
EXAMPLE:
      "Debt ratio %" and "Net worth/Assets" have r = -1.000
      "Debt ratio %" has higher target correlation → KEEP it
      "Net worth/Assets" → DROP it
"""
print("\n── 3.2 Remove Redundant Features ──")

# Calculate correlation matrix
corr_matrix = df[features].corr().abs()

# Find pairs with |r| > 0.95
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# For each highly correlated pair, mark the one with LOWER target correlation
target_corr = df[features].corrwith(df[TARGET]).abs()
to_drop = set()

for col in upper_triangle.columns:
    # Find features that are >0.95 correlated with this column
    high_corr = upper_triangle.index[upper_triangle[col] > 0.95].tolist()
    
    for paired_col in high_corr:
        # Drop the one with LOWER target correlation
        if target_corr[col] >= target_corr[paired_col]:
            to_drop.add(paired_col)
        else:
            to_drop.add(col)

print(f"  Found {len(to_drop)} redundant features to remove:")
for col in sorted(to_drop)[:10]:
    print(f"    ✗ {col}")
if len(to_drop) > 10:
    print(f"    ... and {len(to_drop) - 10} more")

# Drop them
df.drop(columns=list(to_drop), inplace=True)
features = [c for c in df.columns if c != TARGET]
print(f"  ✓ Remaining features: {len(features)}")


# ════════════════════════════════════════════════════════════
# 3.3  FEATURE SELECTION (Mutual Information — Top 30)
# ════════════════════════════════════════════════════════════
"""
WHAT: Keep only the 30 most informative features.

WHY:  "Curse of dimensionality" — too many features relative to 
      samples causes overfitting. With 6,819 rows and 95 features,
      we need to be selective. 30 features is a good balance:
      - Enough to capture diverse business signals
      - Few enough to avoid overfitting
      - Matches the MAX_FEATURES setting in config.py

HOW:  Mutual Information (MI) measures ANY kind of dependency
      between a feature and the target — both linear AND non-linear.
      Unlike correlation, MI catches complex patterns that tree-based
      models will exploit.
      
      We use MI instead of correlation because:
      - Correlation only finds LINEAR relationships
      - MI finds the features most USEFUL for classification
      - Tree models (Random Forest, XGBoost) use non-linear splits
"""
print("\n── 3.3 Feature Selection (Top 30 by Mutual Information) ──")

MAX_FEATURES = 30

X_temp = df[features].copy()
y_temp = df[TARGET].copy()

# Calculate Mutual Information scores
mi_scores = mutual_info_classif(X_temp, y_temp, random_state=RANDOM_STATE, n_neighbors=5)
mi_df = pd.DataFrame({
    'Feature': features,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

# Select top features
selected_features = mi_df.head(MAX_FEATURES)['Feature'].tolist()

print(f"  Top {MAX_FEATURES} features selected by Mutual Information:")
print(f"  {'─' * 55}")
for i, row in mi_df.head(MAX_FEATURES).iterrows():
    print(f"    {mi_df.index.get_loc(i)+1:2d}. MI={row['MI_Score']:.4f}  {row['Feature'][:50]}")

# Keep only selected features + target
df = df[selected_features + [TARGET]]
features = selected_features
print(f"\n  ✓ Dataset reduced to {len(features)} features")


# ════════════════════════════════════════════════════════════
# 3.4  TRAIN / TEST SPLIT (80/20, Stratified)
# ════════════════════════════════════════════════════════════
"""
WHAT: Split data into 80% training and 20% testing sets.

WHY:  We train models on the training set and evaluate on the test
      set (data the model has NEVER seen). This tells us how the
      model will perform on new companies in the real world.

HOW:  "Stratified" split means both sets preserve the 3.2% bankruptcy
      ratio. Without stratification, the small test set might randomly
      get 0 bankrupt companies — making evaluation impossible.

      Olson & Wu used a similar approach: training on 2005-2008 data
      and testing on 2009-2010 data (1,178 train / 143 test).
"""
print("\n── 3.4 Train/Test Split (80/20, Stratified) ──")

X = df[features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,          # 20% for testing
    random_state=RANDOM_STATE,
    stratify=y               # Preserve class ratio in both sets
)

print(f"  Training set: {X_train.shape[0]:,} records ({y_train.mean()*100:.1f}% bankrupt)")
print(f"  Test set:     {X_test.shape[0]:,} records ({y_test.mean()*100:.1f}% bankrupt)")
print(f"  Features:     {X_train.shape[1]}")


# ════════════════════════════════════════════════════════════
# 3.5  SMOTE OVERSAMPLING (Fix Class Imbalance)
# ════════════════════════════════════════════════════════════
"""
WHAT: Generate synthetic bankrupt company examples using SMOTE.

WHY:  With only 3.2% bankrupt cases, most ML models will be biased
      toward predicting "survived" (the majority class). SMOTE fixes
      this by creating synthetic examples of the minority class.

HOW:  SMOTE (Synthetic Minority Over-sampling Technique):
      1. Pick a bankrupt company (sample A)
      2. Find its 5 nearest bankrupt neighbors
      3. Pick one neighbor (sample B)
      4. Create a NEW synthetic sample on the line between A and B
         → synthetic = A + random_fraction × (B - A)
      5. Repeat until classes are balanced

IMPORTANT: We ONLY apply SMOTE to the TRAINING set, never the test set.
           The test set must represent real-world distribution (3.2% bankrupt).

      From Sundararajan's textbook (Ch 11): "Bagging uses resampling 
      technique... compared to a single classifier, bagging improves 
      accuracy and is less affected by noise." SMOTE is a smarter
      version of this resampling idea.
"""
print("\n── 3.5 SMOTE Oversampling ──")

print(f"  Before SMOTE:")
print(f"    Survived:  {(y_train == 0).sum():,}")
print(f"    Bankrupt:  {(y_train == 1).sum():,}")
print(f"    Ratio:     {(y_train == 0).sum() / (y_train == 1).sum():.0f}:1")

smote = SMOTE(
    sampling_strategy='auto',   # Balance to 1:1
    random_state=RANDOM_STATE,
    k_neighbors=5               # 5 nearest neighbors for interpolation
)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\n  After SMOTE:")
print(f"    Survived:  {(y_train_smote == 0).sum():,}")
print(f"    Bankrupt:  {(y_train_smote == 1).sum():,}")
print(f"    Ratio:     {(y_train_smote == 0).sum() / (y_train_smote == 1).sum():.1f}:1")
print(f"    Total:     {len(X_train_smote):,} records (was {len(X_train):,})")


# ════════════════════════════════════════════════════════════
# 3.6  FEATURE SCALING (StandardScaler)
# ════════════════════════════════════════════════════════════
"""
WHAT: Transform each feature to have mean=0 and std=1.

WHY:  Different features have very different scales:
      - "Debt ratio %" ranges 0 to 0.25
      - "Revenue Per Share" ranges 0 to 0.22
      Without scaling, features with larger ranges dominate models
      that use distances (Logistic Regression, Neural Networks, SVM).
      
      Tree-based models (Decision Tree, Random Forest, XGBoost) 
      DON'T need scaling — they split on thresholds, not distances.
      But we scale anyway so all models get the same input.

HOW:  z = (x - mean) / std
      - Fit the scaler on TRAINING data only
      - Transform BOTH train and test with the same scaler
      - This prevents "data leakage" (test info leaking into training)

      From Sundararajan (Ch 12): "We will standardize these features,
      as gradient descent is sensitive to scale."
"""
print("\n── 3.6 Feature Scaling (StandardScaler) ──")

scaler = StandardScaler()

# Fit on training data ONLY, then transform both sets
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_smote),
    columns=features
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=features
)

print(f"  ✓ Scaler fit on training data ({X_train_smote.shape[0]:,} records)")
print(f"  ✓ Training set scaled: mean≈{X_train_scaled.mean().mean():.4f}, std≈{X_train_scaled.std().mean():.4f}")
print(f"  ✓ Test set scaled using SAME scaler (no data leakage)")


# ════════════════════════════════════════════════════════════
# 3.7  SAVE PROCESSED DATA
# ════════════════════════════════════════════════════════════
"""
Save everything we need for Step 4 (Model Training):
- Processed train/test splits (scaled + SMOTE'd)
- Original (unscaled) test set for R analysis
- The scaler object (needed for the prediction app in Step 6)
- Feature list and metadata
"""
print("\n── 3.7 Save Processed Data ──")

os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save scaled datasets (for Python modeling)
X_train_scaled.to_csv('data/processed/X_train.csv', index=False)
pd.DataFrame(y_train_smote).to_csv('data/processed/y_train.csv', index=False)
X_test_scaled.to_csv('data/processed/X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)

# Save unscaled datasets (for R analysis)
X_train_smote.to_csv('data/processed/X_train_unscaled.csv', index=False)
X_test.to_csv('data/processed/X_test_unscaled.csv', index=False)

# Save the scaler (for the prediction app)
joblib.dump(scaler, 'models/scaler.pkl')

# Save feature names and metadata
import json
metadata = {
    'features': features,
    'n_features': len(features),
    'target': TARGET,
    'train_records_original': len(X_train),
    'train_records_after_smote': len(X_train_smote),
    'test_records': len(X_test),
    'bankrupt_pct_test': float(y_test.mean() * 100),
    'random_state': RANDOM_STATE,
    'preprocessing_steps': [
        'Winsorization (1st-99th percentile)',
        'Multicollinearity removal (|r| > 0.95)',
        f'Feature selection (top {MAX_FEATURES} by MI)',
        'Stratified 80/20 train/test split',
        'SMOTE oversampling (training set only)',
        'StandardScaler normalization'
    ]
}

with open('data/processed/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Saved X_train.csv       ({X_train_scaled.shape})")
print(f"  ✓ Saved y_train.csv       ({len(y_train_smote):,} labels)")
print(f"  ✓ Saved X_test.csv        ({X_test_scaled.shape})")
print(f"  ✓ Saved y_test.csv        ({len(y_test):,} labels)")
print(f"  ✓ Saved scaler.pkl        (for prediction app)")
print(f"  ✓ Saved metadata.json     (feature list & settings)")


# ════════════════════════════════════════════════════════════
# 3.8  FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  PREPROCESSING COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
  Raw Input:     6,819 records × 96 columns
  After cleanup: {X_train_scaled.shape[0] + X_test_scaled.shape[0]:,} records × {len(features)} features

  PIPELINE APPLIED:
  ┌─────────────────────────────────────────────────────┐
  │ 1. Winsorized outliers (capped at 1st/99th pct)     │
  │ 2. Removed {len(to_drop):2d} redundant features (|r| > 0.95)      │
  │ 3. Selected top {MAX_FEATURES} features (Mutual Information)    │
  │ 4. Train/Test split: 80/20 stratified               │
  │ 5. SMOTE: {(y_train == 1).sum()} bankrupt → {(y_train_smote == 1).sum():,} synthetic         │
  │ 6. StandardScaler: mean=0, std=1                    │
  └─────────────────────────────────────────────────────┘

  TRAINING SET: {X_train_scaled.shape[0]:,} records (balanced 1:1)
  TEST SET:     {X_test_scaled.shape[0]:,} records (original 3.2% bankrupt)

  → READY FOR STEP 4: MODEL TRAINING
""")
print("=" * 60)

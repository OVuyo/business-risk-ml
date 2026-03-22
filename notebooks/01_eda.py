"""
=============================================================
 STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
 Business Risk ML Project
=============================================================
 This notebook explores the Taiwan Economic Journal bankruptcy
 dataset before we build any models. Think of EDA as a doctor
 examining a patient before prescribing treatment.

 KEY CONCEPTS EXPLAINED:
 - Class imbalance and why it matters
 - Correlation analysis (linear relationships)
 - Mutual Information (non-linear relationships)  
 - Multicollinearity (redundant features)
 - Outlier detection and treatment strategy

 Dataset: 6,819 companies | 95 financial ratios | Target: Bankrupt?
=============================================================
"""

# ── 1. IMPORTS & SETUP ──────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)

# Style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 11,
})


# ── 2. LOAD DATA ────────────────────────────────────────────
df = pd.read_csv('data/data.csv')
df.columns = df.columns.str.strip()

TARGET = 'Bankrupt?'
features = [c for c in df.columns if c != TARGET]

print(f"Dataset: {df.shape[0]:,} records × {df.shape[1]} columns")
print(f"Target:  '{TARGET}'")
print(f"Features: {len(features)}")
print(f"Missing:  {df.isnull().sum().sum()}")
print(f"Dupes:    {df.duplicated().sum()}")


# ── 3. CLASS DISTRIBUTION ───────────────────────────────────
"""
WHY THIS MATTERS:
Only 3.2% of companies went bankrupt. If our model just always 
predicts "survived", it gets 96.8% accuracy — but catches ZERO
bankruptcies. This is called the "accuracy paradox."

SOLUTION (Step 3): We'll use SMOTE (Synthetic Minority Over-sampling)
to generate synthetic bankrupt examples so the model sees a balanced
training set.
"""
print("\nClass Distribution:")
print(df[TARGET].value_counts())
print(f"\nImbalance ratio: {df[TARGET].value_counts()[0] / df[TARGET].value_counts()[1]:.0f}:1")


# ── 4. FEATURE CATEGORIES ───────────────────────────────────
"""
The 95 features map to 6 business dimensions.
This matters because we want our final model to use features
from MULTIPLE dimensions — not just profitability ratios.
A good risk model considers profitability + leverage + liquidity together.
"""
profitability = [c for c in features if any(kw in c.lower() for kw in 
    ['roa', 'roe', 'profit', 'margin', 'earning', 'income', 'eps'])]
liquidity = [c for c in features if any(kw in c.lower() for kw in 
    ['current ratio', 'quick', 'cash', 'working capital', 'liquid', 'no-credit'])]
leverage = [c for c in features if any(kw in c.lower() for kw in 
    ['debt', 'liabilit', 'equity', 'leverage', 'borrow', 'net worth', 'liability'])]
efficiency = [c for c in features if any(kw in c.lower() for kw in 
    ['turnover', 'asset', 'revenue per', 'allocation', 'collection'])]
growth = [c for c in features if 'growth' in c.lower()]

print("\nFeature Categories:")
print(f"  Profitability: {len(profitability)}")
print(f"  Liquidity:     {len(liquidity)}")
print(f"  Leverage:      {len(leverage)}")
print(f"  Efficiency:    {len(efficiency)}")
print(f"  Growth:        {len(growth)}")


# ── 5. CORRELATION ANALYSIS ─────────────────────────────────
"""
KEY CONCEPT: Pearson Correlation
- Measures LINEAR relationship between two variables
- Range: -1 (perfect negative) to +1 (perfect positive)
- LIMITATION: misses non-linear patterns (that's why we also use MI)

FINDINGS:
- Net Income to Total Assets has the strongest correlation (-0.316)
- Negative correlation = LOWER values → MORE bankruptcy risk
- Positive correlation = HIGHER values → MORE bankruptcy risk
- The ROA variants are highly correlated WITH EACH OTHER (multicollinearity)
"""
corr_target = df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("\nTop 10 Features by Correlation with Bankruptcy:")
for i, (col, val) in enumerate(corr_target.head(10).items(), 1):
    direction = "↑ risk" if val > 0 else "↓ risk"
    print(f"  {i:2d}. r={val:+.3f} ({direction})  {col}")


# ── 6. MULTICOLLINEARITY DETECTION ──────────────────────────
"""
KEY CONCEPT: Multicollinearity
When two features are very highly correlated (|r| > 0.9), they carry
nearly the same information. Including both:
- Wastes computation
- Can confuse some models (especially logistic regression)
- Makes feature importance unreliable

We found 28 pairs with |r| > 0.9. In Step 3, we'll drop redundant features.

EXAMPLE: "Debt ratio %" and "Net worth/Assets" have r = -1.000
         They are literally the SAME information (Debt% = 1 - NetWorth%)
         We only need one of them.
"""
corr_full = df[features].corr()
high_corr = []
for i in range(len(corr_full.columns)):
    for j in range(i+1, len(corr_full.columns)):
        if abs(corr_full.iloc[i, j]) > 0.9:
            high_corr.append((corr_full.columns[i], corr_full.columns[j], 
                             round(corr_full.iloc[i, j], 3)))

print(f"\nHighly Correlated Pairs (|r| > 0.9): {len(high_corr)}")
for f1, f2, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:5]:
    print(f"  r={r:+.3f}: {f1[:40]} ↔ {f2[:40]}")


# ── 7. MUTUAL INFORMATION (NON-LINEAR IMPORTANCE) ───────────
"""
KEY CONCEPT: Mutual Information (MI)
- Measures ANY kind of dependency (linear AND non-linear)
- If MI = 0, features are completely independent
- Higher MI = more useful for prediction

COMPARE with Correlation:
- Correlation says "Net Income to Total Assets" is #1
- MI says "Persistent EPS in the Last Four Seasons" is #1
- Both are valuable perspectives — they catch different patterns

This is exactly why Olson & Wu recommend running MULTIPLE models:
different models capture different patterns in the data.
"""
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Cap outliers before MI
for col in X.columns:
    X[col] = X[col].clip(X[col].quantile(0.01), X[col].quantile(0.99))

mi = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI': mi}).sort_values('MI', ascending=False)

print("\nTop 10 Features by Mutual Information:")
for i, row in mi_df.head(10).iterrows():
    print(f"  MI={row['MI']:.4f}  {row['Feature']}")


# ── 8. OUTLIER ANALYSIS ─────────────────────────────────────
"""
KEY CONCEPT: Outliers
13 features have max values in the BILLIONS while their 99th percentile
is below 1.0. These are clearly data entry errors.

STRATEGY: Cap at the 99th percentile (called "Winsorization")
- Preserves the data point (doesn't delete the row)
- Removes the extreme value's distortion
- Standard practice in financial data analysis
"""
df_raw = pd.read_csv('data/data.csv')
df_raw.columns = df_raw.columns.str.strip()
outlier_cols = []
for col in features:
    mx = df_raw[col].max()
    q99 = df_raw[col].quantile(0.99)
    if mx > 10 * q99 and mx > 10:
        outlier_cols.append(col)
        
print(f"\nFeatures with Extreme Outliers: {len(outlier_cols)}")
for col in outlier_cols:
    n = (df_raw[col] > df_raw[col].quantile(0.99) * 10).sum()
    print(f"  {col[:50]:50s}  ({n} extreme rows)")


# ── 9. KEY FINDINGS SUMMARY ─────────────────────────────────
print("\n" + "="*60)
print("  EDA SUMMARY — KEY FINDINGS")
print("="*60)
print("""
  1. CLEAN DATA: Zero missing values, zero duplicates
  
  2. SEVERE IMBALANCE: 96.8% survived vs 3.2% bankrupt (30:1)
     → Must use SMOTE or class weights in modeling
  
  3. TOP PREDICTORS (by correlation):
     • Net Income to Total Assets (r=-0.316)
     • ROA variants (r≈-0.27)
     • Debt ratio % (r=+0.250)
  
  4. TOP PREDICTORS (by mutual information):
     • Persistent EPS (MI=0.043)
     • Net profit before tax/Capital (MI=0.039)
     • Net Income to Stockholder's Equity (MI=0.039)
  
  5. MULTICOLLINEARITY: 28 feature pairs with |r|>0.9
     → Must remove redundant features
  
  6. OUTLIERS: 13 features with billion-level max values
     → Will cap at 99th percentile
  
  7. BUSINESS INSIGHT: Bankruptcy is predicted by a 
     COMBINATION of declining profitability + rising debt.
     No single feature is strongly predictive alone.
""")
print("  → NEXT: Step 3 (Preprocessing) will fix all these issues")
print("="*60)

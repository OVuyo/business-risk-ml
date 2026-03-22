"""
Business Risk ML - Preprocessing Module
========================================
Reusable functions for the preprocessing pipeline.
Called by notebooks/02_preprocessing.py and app/streamlit_app.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE


def winsorize(df: pd.DataFrame, features: list, 
              lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """
    Cap extreme values at specified percentiles.
    
    Parameters
    ----------
    df : DataFrame with features to winsorize
    features : list of column names to process
    lower_pct : lower percentile bound (default 1st)
    upper_pct : upper percentile bound (default 99th)
    
    Returns
    -------
    DataFrame with capped values
    """
    df = df.copy()
    for col in features:
        q_low = df[col].quantile(lower_pct)
        q_high = df[col].quantile(upper_pct)
        df[col] = df[col].clip(lower=q_low, upper=q_high)
    return df


def remove_multicollinear(df: pd.DataFrame, features: list, 
                          target: str, threshold: float = 0.95) -> list:
    """
    Remove one feature from each highly correlated pair.
    Keeps the feature with higher target correlation.
    
    Parameters
    ----------
    df : DataFrame
    features : list of feature column names
    target : target column name
    threshold : correlation threshold (default 0.95)
    
    Returns
    -------
    list of features to DROP
    """
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    target_corr = df[features].corrwith(df[target]).abs()
    
    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        for paired in high_corr:
            if target_corr[col] >= target_corr[paired]:
                to_drop.add(paired)
            else:
                to_drop.add(col)
    
    return list(to_drop)


def select_features_mi(X: pd.DataFrame, y: pd.Series, 
                       n_features: int = 30, random_state: int = 42) -> list:
    """
    Select top features by Mutual Information score.
    
    Parameters
    ----------
    X : feature DataFrame
    y : target Series
    n_features : number of features to keep
    random_state : for reproducibility
    
    Returns
    -------
    list of selected feature names
    """
    mi_scores = mutual_info_classif(X, y, random_state=random_state, n_neighbors=5)
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)
    
    return mi_df.head(n_features)['Feature'].tolist()


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series,
                random_state: int = 42) -> tuple:
    """
    Balance classes using SMOTE oversampling.
    
    Parameters
    ----------
    X_train : training features
    y_train : training labels
    random_state : for reproducibility
    
    Returns
    -------
    tuple of (X_resampled, y_resampled)
    """
    smote = SMOTE(sampling_strategy='auto', random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def full_pipeline(filepath: str, target: str = 'Bankrupt?',
                  max_features: int = 30, test_size: float = 0.20,
                  random_state: int = 42) -> dict:
    """
    Run the complete preprocessing pipeline.
    
    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, scaler, features
    """
    # Load
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    features = [c for c in df.columns if c != target]
    
    # 1. Winsorize
    df = winsorize(df, features)
    
    # 2. Remove multicollinear
    to_drop = remove_multicollinear(df, features, target)
    df.drop(columns=to_drop, inplace=True)
    features = [c for c in df.columns if c != target]
    
    # 3. Feature selection
    selected = select_features_mi(df[features], df[target], max_features, random_state)
    df = df[selected + [target]]
    features = selected
    
    # 4. Train/test split
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 5. SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, random_state)
    
    # 6. Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_smote), columns=features
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=features
    )
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_smote,
        'y_test': y_test,
        'scaler': scaler,
        'features': features,
    }

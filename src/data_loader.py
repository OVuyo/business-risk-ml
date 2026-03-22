"""
Business Risk ML - Data Loader & Validator
==========================================
This module handles loading the Kaggle bankruptcy dataset,
performing initial validation, and preparing it for analysis.

Dataset: Taiwan Economic Journal Company Bankruptcy Prediction
Source:   https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction

Key Concepts:
- The dataset contains 6,819 company records with 95 financial features
- Target variable: 'Bankrupt?' (1 = bankrupt, 0 = survived)
- This is an IMBALANCED dataset (~3% bankrupt) — we'll handle this in preprocessing
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the bankruptcy dataset from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file downloaded from Kaggle
    
    Returns
    -------
    pd.DataFrame
        Loaded and initially cleaned DataFrame
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Clean column names: strip whitespace
    df.columns = df.columns.str.strip()
    
    print(f"  ✓ Loaded {df.shape[0]:,} records with {df.shape[1]} features")
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Run validation checks on the loaded dataset.
    Returns a report dictionary with findings.
    
    This is important because real-world data often has:
    - Missing values (NaN)
    - Duplicate rows
    - Incorrect data types
    - Outliers that could skew our models
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_records": len(df),
        "total_features": len(df.columns),
        "target_column": None,
        "issues": [],
        "warnings": [],
    }
    
    # 1. Identify the target column
    target_candidates = ['Bankrupt?', 'bankruptcy', 'Bankrupt', 'target', 'class']
    for col in target_candidates:
        if col in df.columns:
            report["target_column"] = col
            break
    
    if report["target_column"] is None:
        report["issues"].append("Could not identify target column!")
    else:
        target = report["target_column"]
        counts = df[target].value_counts()
        report["class_distribution"] = counts.to_dict()
        
        # Check for class imbalance
        minority_pct = counts.min() / counts.sum() * 100
        if minority_pct < 10:
            report["warnings"].append(
                f"Severe class imbalance detected: minority class is only {minority_pct:.1f}%"
            )
    
    # 2. Check for missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        report["missing_values"] = {
            col: int(count) for col, count in missing_cols.items()
        }
        report["warnings"].append(
            f"{len(missing_cols)} columns have missing values"
        )
    else:
        report["missing_values"] = {}
    
    # 3. Check for duplicates
    n_duplicates = df.duplicated().sum()
    report["duplicate_rows"] = int(n_duplicates)
    if n_duplicates > 0:
        report["warnings"].append(f"{n_duplicates} duplicate rows found")
    
    # 4. Check data types
    report["dtypes"] = {
        "numeric": len(df.select_dtypes(include=[np.number]).columns),
        "categorical": len(df.select_dtypes(include=['object', 'category']).columns),
    }
    
    # 5. Basic statistics summary
    report["numeric_summary"] = {
        "min_values_negative": int((df.select_dtypes(include=[np.number]).min() < 0).sum()),
        "columns_all_zero": int((df.select_dtypes(include=[np.number]).sum() == 0).sum()),
    }
    
    return report


def print_validation_report(report: dict):
    """Pretty-print the validation report."""
    print("\n" + "=" * 60)
    print("  DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"  Records:  {report['total_records']:,}")
    print(f"  Features: {report['total_features']}")
    print(f"  Target:   {report['target_column']}")
    
    if "class_distribution" in report:
        print(f"\n  Class Distribution:")
        for label, count in report["class_distribution"].items():
            pct = count / report["total_records"] * 100
            print(f"    {label}: {count:,} ({pct:.1f}%)")
    
    print(f"\n  Data Types:")
    print(f"    Numeric:     {report['dtypes']['numeric']}")
    print(f"    Categorical: {report['dtypes']['categorical']}")
    
    print(f"\n  Missing Values: {len(report['missing_values'])} columns affected")
    print(f"  Duplicate Rows: {report['duplicate_rows']}")
    
    if report["warnings"]:
        print(f"\n  ⚠  Warnings:")
        for w in report["warnings"]:
            print(f"    • {w}")
    
    if report["issues"]:
        print(f"\n  ✗  Issues:")
        for i in report["issues"]:
            print(f"    • {i}")
    
    if not report["issues"]:
        print(f"\n  ✓  Data passed all validation checks!")
    print("=" * 60 + "\n")


def get_feature_groups(df: pd.DataFrame, target_col: str) -> dict:
    """
    Group features by category for better understanding.
    
    The Kaggle Taiwan dataset uses financial ratio names.
    This function maps them into intuitive business categories
    so we can understand WHAT we're predicting WITH.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Common financial ratio keyword mapping
    profitability = [c for c in feature_cols if any(
        kw in c.lower() for kw in ['roa', 'roe', 'profit', 'margin', 'earning', 'income']
    )]
    
    liquidity = [c for c in feature_cols if any(
        kw in c.lower() for kw in ['current', 'quick', 'cash', 'working capital', 'liquid']
    )]
    
    leverage = [c for c in feature_cols if any(
        kw in c.lower() for kw in ['debt', 'liabilit', 'equity', 'leverage', 'borrow']
    )]
    
    efficiency = [c for c in feature_cols if any(
        kw in c.lower() for kw in ['turnover', 'asset', 'inventor', 'revenue', 'sales']
    )]
    
    # Everything else
    assigned = set(profitability + liquidity + leverage + efficiency)
    other = [c for c in feature_cols if c not in assigned]
    
    groups = {
        "profitability": profitability,
        "liquidity": liquidity,
        "leverage": leverage,
        "efficiency": efficiency,
        "other": other,
    }
    
    print("\nFeature Groups:")
    for name, cols in groups.items():
        print(f"  {name.title():15s}: {len(cols)} features")
    
    return groups


# ── Main execution (when run as script) ──────────────────────
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <path_to_csv>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # Load
    df = load_data(filepath)
    
    # Validate
    report = validate_data(df)
    print_validation_report(report)
    
    # Group features
    if report["target_column"]:
        groups = get_feature_groups(df, report["target_column"])
    
    # Save report
    report_path = os.path.join(os.path.dirname(filepath), "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to: {report_path}")

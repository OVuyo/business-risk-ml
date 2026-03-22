"""
=============================================================
 STEP 6: STREAMLIT PREDICTION APP
 Business Risk ML Project
=============================================================
 A web interface where users can input financial ratios and
 get a bankruptcy risk prediction from our trained models.

 Run with: streamlit run app/streamlit_app.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Business Bankruptcy Risk Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load Model & Metadata ───────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)
    with open('models/best_model_name.txt', 'r') as f:
        model_name = f.read().strip()
    return model, scaler, metadata, model_name

try:
    model, scaler, metadata, model_name = load_model()
    features = metadata['features']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Run notebooks/02_preprocessing.py and 03_model_training.py first.")
    st.stop()


# ── Header ───────────────────────────────────────────────────
st.title("📊 Business Bankruptcy Risk Predictor")
st.markdown(f"""
Powered by **{model_name}** trained on 6,819 companies from the 
Taiwan Economic Journal dataset. Enter financial ratios below to predict
bankruptcy risk.
""")

st.divider()

# ── Sidebar: About ───────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Model Performance:**
    - Based on 30 financial ratio features
    - Trained with SMOTE-balanced data
    - Evaluated on held-out test set
    
    **How to use:**
    1. Enter financial ratios in the form
    2. Click "Predict Risk"
    3. View the prediction and confidence
    
    **Feature Categories:**
    - 🟢 Profitability (ROA, Net Income)
    - 🔴 Leverage (Debt ratio, Borrowing)
    - 🔵 Liquidity (Current ratio, Cash)
    - 🟡 Efficiency (Turnover rates)
    """)
    
    st.divider()
    st.caption("Business Risk ML Project")
    st.caption("Based on Olson & Wu (2017) and Sundararajan (2025)")


# ── Feature Descriptions ─────────────────────────────────────
feature_info = {
    'Persistent EPS in the Last Four Seasons': ('Profitability', 0.1, 0.4, 0.22),
    'Net Income to Stockholder\'s Equity': ('Profitability', 0.0, 1.0, 0.80),
    'Net Income to Total Assets': ('Profitability', 0.5, 1.0, 0.80),
    'Net Value Growth Rate': ('Growth', -0.5, 1.0, 0.0),
    'Borrowing dependency': ('Leverage', 0.0, 0.5, 0.37),
    'Interest Expense Ratio': ('Leverage', 0.0, 0.05, 0.01),
    'Continuous interest rate (after tax)': ('Profitability', 0.5, 1.0, 0.78),
    'Total debt/Total net worth': ('Leverage', 0.0, 0.5, 0.02),
    'Retained Earnings to Total Assets': ('Profitability', 0.7, 1.0, 0.93),
    'Degree of Financial Leverage (DFL)': ('Leverage', 0.0, 1.0, 0.80),
    'Equity to Liability': ('Leverage', 0.0, 1.0, 0.80),
    'Net worth/Assets': ('Leverage', 0.5, 1.0, 0.87),
    'Total income/Total expense': ('Profitability', 0.9, 1.1, 1.00),
    'Interest Coverage Ratio (Interest expense to EBIT)': ('Leverage', 0.0, 0.2, 0.03),
    'Net Value Per Share (A)': ('Valuation', 0.0, 0.5, 0.22),
    'Current Liability to Current Assets': ('Liquidity', 0.0, 0.5, 0.12),
    'Non-industry income and expenditure/revenue': ('Profitability', 0.0, 0.5, 0.30),
    'Current Ratio': ('Liquidity', 0.0, 0.1, 0.03),
    'Working Capital to Total Assets': ('Liquidity', -0.2, 0.5, 0.14),
    'Current Liability to Equity': ('Leverage', 0.0, 0.3, 0.08),
    'Operating Profit Per Share (Yuan ¥)': ('Profitability', 0.0, 0.3, 0.14),
    'Working Capital/Equity': ('Liquidity', -0.2, 0.5, 0.15),
    'Operating profit per person': ('Efficiency', 0.0, 0.5, 0.10),
    'Cash/Current Liability': ('Liquidity', 0.0, 0.5, 0.08),
    'Inventory/Working Capital': ('Efficiency', 0.0, 0.5, 0.10),
    'Tax rate (A)': ('Other', 0.0, 1.0, 0.20),
    'Working capitcal Turnover Rate': ('Efficiency', 0.0, 0.5, 0.04),
    'Operating Profit Rate': ('Profitability', 0.95, 1.05, 1.00),
    'Current Liability to Assets': ('Leverage', 0.0, 0.3, 0.10),
    'Gross Profit to Sales': ('Profitability', 0.0, 1.0, 0.60),
}


# ── Input Form ───────────────────────────────────────────────
st.subheader("Enter Financial Ratios")

# Quick-fill options
col_preset1, col_preset2, col_preset3 = st.columns(3)
with col_preset1:
    if st.button("📗 Load Healthy Company Example"):
        st.session_state['preset'] = 'healthy'
with col_preset2:
    if st.button("📕 Load At-Risk Company Example"):
        st.session_state['preset'] = 'risky'
with col_preset3:
    if st.button("🔄 Reset to Defaults"):
        st.session_state['preset'] = 'default'

# Organize features into tabs by category
tabs = st.tabs(["🟢 Profitability", "🔴 Leverage", "🔵 Liquidity", "🟡 Efficiency & Other"])
input_values = {}

for feat in features:
    info = feature_info.get(feat, ('Other', 0.0, 1.0, 0.5))
    category, min_val, max_val, default_val = info
    
    # Adjust defaults based on preset
    if st.session_state.get('preset') == 'healthy':
        if category == 'Profitability':
            default_val = max_val * 0.8
        elif category == 'Leverage':
            default_val = min_val + (max_val - min_val) * 0.2
    elif st.session_state.get('preset') == 'risky':
        if category == 'Profitability':
            default_val = min_val + (max_val - min_val) * 0.2
        elif category == 'Leverage':
            default_val = max_val * 0.8
    
    # Pick the right tab
    if category == 'Profitability':
        tab = tabs[0]
    elif category == 'Leverage':
        tab = tabs[1]
    elif category in ('Liquidity',):
        tab = tabs[2]
    else:
        tab = tabs[3]
    
    with tab:
        input_values[feat] = st.slider(
            feat,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            step=0.001,
            format="%.4f"
        )


# ── Prediction ───────────────────────────────────────────────
st.divider()

if st.button("🔮 Predict Bankruptcy Risk", type="primary", use_container_width=True):
    # Prepare input
    input_df = pd.DataFrame([input_values])[features]
    
    # Scale using the same scaler from training
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    prob_bankrupt = probability[1]
    prob_survived = probability[0]
    
    # Display result
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if prediction == 1:
            st.error("⚠️ HIGH BANKRUPTCY RISK")
            st.metric("Bankruptcy Probability", f"{prob_bankrupt:.1%}")
        else:
            st.success("✅ LOW BANKRUPTCY RISK")
            st.metric("Survival Probability", f"{prob_survived:.1%}")
        
        # Confidence bar
        st.progress(prob_survived, text=f"Survival confidence: {prob_survived:.1%}")
        
        # Risk factors
        st.subheader("Key Risk Factors")
        input_series = pd.Series(input_values)
        
        # Compare to typical values
        risk_factors = []
        if input_values.get('Borrowing dependency', 0) > 0.40:
            risk_factors.append("🔴 High borrowing dependency")
        if input_values.get('Net Income to Total Assets', 1) < 0.75:
            risk_factors.append("🔴 Low net income to assets ratio")
        if input_values.get('Retained Earnings to Total Assets', 1) < 0.88:
            risk_factors.append("🟡 Below-average retained earnings")
        if input_values.get('Current Ratio', 1) < 0.02:
            risk_factors.append("🟡 Low current ratio (liquidity concern)")
        if input_values.get('Persistent EPS in the Last Four Seasons', 1) < 0.19:
            risk_factors.append("🔴 Declining earnings per share")
        
        if risk_factors:
            for rf in risk_factors:
                st.markdown(f"- {rf}")
        else:
            st.markdown("No major risk factors detected.")

    st.divider()
    st.caption(f"Model: {model_name} | Features: {len(features)} | Dataset: Taiwan Economic Journal")


# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
Business Risk ML Project | Based on Olson & Wu (2017) Enterprise Risk Management Models
and Sundararajan (2025) Multivariate Analysis and Machine Learning Techniques
</div>
""", unsafe_allow_html=True)

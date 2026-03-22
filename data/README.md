# Data Directory

## How to get the dataset

This project uses the **Company Bankruptcy Prediction** dataset from Kaggle.

### Download Instructions

1. Go to: https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction
2. Click **Download** (requires free Kaggle account)
3. Extract the ZIP file
4. Place `data.csv` in this directory

### Dataset Overview

| Property       | Value                                     |
|----------------|-------------------------------------------|
| Source         | Taiwan Economic Journal (1999–2009)        |
| Records        | 6,819 companies                           |
| Features       | 95 financial ratios                       |
| Target         | `Bankrupt?` (1 = bankrupt, 0 = survived)  |
| Class Balance  | ~3% bankrupt (imbalanced)                 |

### Key Features Include

- **ROA** — Return on Assets (profitability)
- **Debt Ratio** — Total Liabilities / Total Assets (leverage)
- **Current Ratio** — Current Assets / Current Liabilities (liquidity)
- **Net Income** — Bottom line earnings
- **Cash Flow** — Operating cash flow metrics
- **Revenue** — Sales and turnover metrics

### Academic Reference

This dataset aligns with the bankruptcy prediction approach described in:
- Olson, D.L., Delen, D., and Meng, Y. (2012). *Comparative analysis of data mining methods for bankruptcy prediction*. Decision Support Systems, 52(2), 464–473.

### Note

The raw CSV is excluded from git (see `.gitignore`). Each collaborator must download their own copy.

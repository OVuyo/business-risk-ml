"""
Business Risk ML - Project Configuration
=========================================
Central place for all project settings.
Change values here instead of hunting through notebooks.
"""

import os

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# ── Data Settings ────────────────────────────────────────────
RAW_DATA_FILE = "data.csv"              # Kaggle download filename
TARGET_COLUMN = "Bankrupt?"             # Binary target: 1=bankrupt, 0=survived
TEST_SIZE = 0.20                        # 80/20 train-test split
VALIDATION_SIZE = 0.15                  # 15% of training for validation
RANDOM_STATE = 42                       # For reproducibility

# ── Model Settings ───────────────────────────────────────────
MODELS_TO_TRAIN = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "neural_network",
]

# ── Class Imbalance Strategy ─────────────────────────────────
# The Kaggle dataset is heavily imbalanced (~3% bankrupt)
# Options: "smote", "class_weight", "undersample", "none"
IMBALANCE_STRATEGY = "smote"

# ── Feature Selection ────────────────────────────────────────
# Maximum features to keep after selection (0 = keep all)
MAX_FEATURES = 30

# ── Neural Network ───────────────────────────────────────────
NN_EPOCHS = 100
NN_BATCH_SIZE = 32
NN_HIDDEN_LAYERS = [64, 32, 16]
NN_LEARNING_RATE = 0.001
NN_EARLY_STOPPING_PATIENCE = 10

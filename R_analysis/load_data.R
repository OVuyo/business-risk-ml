# ============================================================
# Business Risk ML - R Data Loading Script
# ============================================================
# This script loads the preprocessed data from Python's pipeline
# so R can perform its own statistical analysis and modeling.
#
# Run AFTER: notebooks/02_preprocessing.py
# ============================================================

library(readr)
library(dplyr)

cat("Loading preprocessed data from Python pipeline...\n\n")

# ── Load Data ────────────────────────────────────────────────
X_train <- read_csv("data/processed/X_train_unscaled.csv", show_col_types = FALSE)
y_train <- read_csv("data/processed/y_train.csv", show_col_types = FALSE)[[1]]
X_test  <- read_csv("data/processed/X_test_unscaled.csv", show_col_types = FALSE)
y_test  <- read_csv("data/processed/y_test.csv", show_col_types = FALSE)[[1]]

# Combine features + target for R modeling
train_df <- X_train %>% mutate(Bankrupt = as.factor(y_train))
test_df  <- X_test  %>% mutate(Bankrupt = as.factor(y_test))

cat(sprintf("  Training set: %d records (%d features)\n", nrow(train_df), ncol(train_df) - 1))
cat(sprintf("  Test set:     %d records (%d features)\n", nrow(test_df), ncol(test_df) - 1))
cat(sprintf("  Train class balance: %d survived / %d bankrupt\n",
            sum(y_train == 0), sum(y_train == 1)))
cat(sprintf("  Test class balance:  %d survived / %d bankrupt\n",
            sum(y_test == 0), sum(y_test == 1)))

# ── Clean Column Names for R ─────────────────────────────────
# R doesn't like spaces and special chars in column names
clean_names <- function(df) {
  names(df) <- make.names(names(df), unique = TRUE)
  return(df)
}

train_df <- clean_names(train_df)
test_df  <- clean_names(test_df)

cat("\n  ✓ Data loaded and ready for R analysis!\n")
cat("  ✓ Use train_df and test_df for modeling\n")
cat("  ✓ Target column: 'Bankrupt' (factor: 0 or 1)\n\n")

# ── Quick Summary ─────────────────────────────────────────────
cat("Feature names:\n")
cat(paste("  ", names(train_df)[1:10], collapse = "\n"), "\n")
if (ncol(train_df) > 11) {
  cat(sprintf("  ... and %d more\n", ncol(train_df) - 11))
}

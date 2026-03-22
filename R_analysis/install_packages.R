# ============================================================
# Business Risk ML - R Package Installation
# Run this script once to install all required R packages
# ============================================================

# CRAN packages
required_packages <- c(
  # Data manipulation
  "dplyr",
  "tidyr",
  "readr",
  
  # Visualization
  "ggplot2",
  "corrplot",
  "gridExtra",
  "scales",
  "RColorBrewer",
  
  # Machine Learning & Statistics
  "caret",
  "randomForest",
  "rpart",
  "rpart.plot",
  "e1071",          # SVM + Naive Bayes
  "glmnet",         # Regularized regression
  
  # Model Evaluation
  "pROC",           # ROC curves & AUC
  "ROCR",           # Prediction performance
  
  # Reporting
  "knitr",
  "rmarkdown"
)

# Install missing packages
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste0("Installing: ", pkg, "\n"))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    cat(paste0("Already installed: ", pkg, "\n"))
  }
}

invisible(sapply(required_packages, install_if_missing))

cat("\n✓ All R packages are ready!\n")

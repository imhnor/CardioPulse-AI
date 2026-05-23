"""
run_all.py
----------
Master launcher -- runs the FULL pipeline end to end:

  Step 1: Preprocessing       (skipped if preprocessed/ exists)
  Step 2: Logistic Regression (tabular baseline)
  Step 3: Random Forest       (tabular, strong baseline)
  Step 4: XGBoost             (pip install xgboost)
  Step 5: CNN-1D              (raw ECG signals, requires torch)
  Step 6: ResNet-1D           (raw ECG signals, requires torch)
  Step 7: CNN-BiLSTM          (raw ECG signals, requires torch)
  Step 8: Evaluate ALL        (comparison table)

Usage:
  $env:PYTHONIOENCODING='utf-8'
  .\\ecg_env\\Scripts\\python.exe run_all.py

Skip a step by deleting / keeping the model file in models/.
Force re-train of a model by deleting its .pkl / .pth file.
Force re-preprocessing by deleting the preprocessed/ folder.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  ECG MULTI-LABEL CLASSIFICATION -- FULL PIPELINE")
print("=" * 60)

# ------------------------------------------------------------------
# Step 1: Preprocessing
# ------------------------------------------------------------------
print("\n[STEP 1] Preprocessing")
print("-" * 40)
from pipelines.preprocessing_pipeline import run as preprocess
preprocess()

# ------------------------------------------------------------------
# Step 2: Logistic Regression (fast baseline)
# ------------------------------------------------------------------
print("\n[STEP 2] Logistic Regression (tabular baseline)")
print("-" * 40)
from pipelines.train_logistic_regression import main as train_lr
train_lr()

# ------------------------------------------------------------------
# Step 3: Random Forest
# ------------------------------------------------------------------
print("\n[STEP 3] Random Forest")
print("-" * 40)
from pipelines.train_random_forest import main as train_rf
train_rf()

# ------------------------------------------------------------------
# Step 4: XGBoost (optional -- pip install xgboost)
# ------------------------------------------------------------------
print("\n[STEP 4] XGBoost")
print("-" * 40)
try:
    from pipelines.train_xgboost import main as train_xgb
    train_xgb()
except SystemExit:
    print("  [SKIP] XGBoost -- not installed (pip install xgboost)")

# ------------------------------------------------------------------
# Step 5: CNN-1D (DL -- raw ECG signals)
# ------------------------------------------------------------------
print("\n[STEP 5] CNN-1D (Deep Learning -- raw ECG)")
print("-" * 40)
try:
    from pipelines.train_cnn1d import main as train_cnn1d
    train_cnn1d()
except SystemExit:
    print("  [SKIP] CNN-1D -- PyTorch not installed (pip install torch)")

# ------------------------------------------------------------------
# Step 6: ResNet-1D
# ------------------------------------------------------------------
print("\n[STEP 6] ResNet-1D (Deep Learning -- raw ECG)")
print("-" * 40)
try:
    from pipelines.train_resnet1d import main as train_resnet
    train_resnet()
except SystemExit:
    print("  [SKIP] ResNet-1D -- PyTorch not installed")

# ------------------------------------------------------------------
# Step 7: CNN-BiLSTM
# ------------------------------------------------------------------
print("\n[STEP 7] CNN-BiLSTM (Deep Learning -- raw ECG)")
print("-" * 40)
try:
    from pipelines.train_cnn_lstm import main as train_cnnlstm
    train_cnnlstm()
except SystemExit:
    print("  [SKIP] CNN-BiLSTM -- PyTorch not installed")

# ------------------------------------------------------------------
# Step 8: Evaluate ALL
# ------------------------------------------------------------------
print("\n[STEP 8] Evaluating All Models")
print("-" * 40)
from evaluate_all import main as evaluate
evaluate()

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("  Models saved in : models/")
print("  Run evaluate_all.py anytime to compare models")
print("=" * 60)

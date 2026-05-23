"""
config.py — Shared Configuration for ECG ML Pipelines
All paths and hyper-parameter defaults live here so every
pipeline file stays in sync automatically.
"""

import os

# ── Root of the repo ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Raw dataset (PTB-XL) ──────────────────────────────────────────────────────
DATASET_DIR   = os.path.join(BASE_DIR, "C:\Users\letsl\Documents\GitHub\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
RAW_CSV       = os.path.join(DATASET_DIR, "ptbxl_database.csv")
SCP_CSV       = os.path.join(DATASET_DIR, "scp_statements.csv")

# ── Preprocessed artefacts ────────────────────────────────────────────────────
PREPROCESSED_DIR   = os.path.join(BASE_DIR, "preprocessed")
CLEANED_CSV        = os.path.join(PREPROCESSED_DIR, "ptbxl_cleaned.csv")
SCALER_FILE        = os.path.join(PREPROCESSED_DIR, "scaler.pkl")
LABEL_BINARIZER    = os.path.join(PREPROCESSED_DIR, "mlb.pkl")
TRAIN_CSV          = os.path.join(PREPROCESSED_DIR, "train.csv")
VAL_CSV            = os.path.join(PREPROCESSED_DIR, "val.csv")
TEST_CSV           = os.path.join(PREPROCESSED_DIR, "test.csv")

# ── Saved model directory ─────────────────────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Target labels (superclasses) ──────────────────────────────────────────────
TARGET_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

# ── Feature columns used for training ────────────────────────────────────────
FEATURE_COLS = ["age", "sex", "height", "weight", "pacemaker"]

# ── PTB-XL stratified-fold split (fold 10 = test, fold 9 = val) ──────────────
TEST_FOLD = 10
VAL_FOLD  = 9

# ── Misc ──────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42

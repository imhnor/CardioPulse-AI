"""
train_random_forest.py
----------------------
Trains a Random Forest multi-label classifier on the ECG tabular data.

Steps:
  1. Runs preprocessing_pipeline (skips if already done)
  2. Loads train / val / test splits
  3. Trains RandomForestClassifier with class_weight handling
  4. Evaluates on val + test sets
  5. Saves model to models/random_forest.pkl

Usage:
  python pipelines/train_random_forest.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg
from pipelines.preprocessing_pipeline import run as preprocess

# ------------------------------------------------------------
MODEL_PATH = os.path.join(cfg.MODELS_DIR, "random_forest.pkl")
# ------------------------------------------------------------


def load_splits():
    train = pd.read_csv(cfg.TRAIN_CSV)
    val   = pd.read_csv(cfg.VAL_CSV)
    test  = pd.read_csv(cfg.TEST_CSV)
    return train, val, test


def evaluate(model, X, y_true, split_name: str):
    y_pred = model.predict(X)
    hl     = hamming_loss(y_true, y_pred)
    macro  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n-- {split_name} --")
    print(f"  Hamming Loss : {hl:.4f}")
    print(f"  Macro F1     : {macro:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=cfg.TARGET_LABELS,
                                zero_division=0))


def main():
    # -- Step 1: Ensure preprocessing is done ---------------------------------
    preprocess()

    # -- Step 2: Check if model already exists ---------------------------------
    if os.path.exists(MODEL_PATH):
        print(f"[OK] Model already exists at {MODEL_PATH}")
        print("   Skipping training. Delete the file to retrain.\n")
        ans = input("   Re-train anyway? [y/N]: ").strip().lower()
        if ans != "y":
            return

    # -- Step 3: Load data -----------------------------------------------------
    print("\n[1/3] Loading splits ...")
    train, val, test = load_splits()

    X_train = train[cfg.FEATURE_COLS].values
    y_train = train[cfg.TARGET_LABELS].values
    X_val   = val[cfg.FEATURE_COLS].values
    y_val   = val[cfg.TARGET_LABELS].values
    X_test  = test[cfg.FEATURE_COLS].values
    y_test  = test[cfg.TARGET_LABELS].values

    # -- Step 4: Train ---------------------------------------------------------
    print("[2/3] Training Random Forest ...")
    base_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=cfg.RANDOM_STATE,
        verbose=1,
    )
    model = MultiOutputClassifier(base_rf, n_jobs=-1)
    model.fit(X_train, y_train)

    # -- Step 5: Evaluate ------------------------------------------------------
    print("[3/3] Evaluating ...")
    evaluate(model, X_val,  y_val,  "Validation")
    evaluate(model, X_test, y_test, "Test")

    # -- Save ------------------------------------------------------------
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[OK] Model saved -> {MODEL_PATH}\n")


if __name__ == "__main__":
    main()

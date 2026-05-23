"""
train_xgboost.py
----------------
Trains XGBoost (one-vs-rest per label) on the ECG tabular data.

Steps:
  1. Runs preprocessing_pipeline (skips if already done)
  2. Loads train / val / test splits
  3. Trains XGBClassifier per label (MultiOutputClassifier wrapper)
  4. Evaluates on val + test sets
  5. Saves model to models/xgboost.pkl

Usage:
  python pipelines/train_xgboost.py

Note:
  Install XGBoost if not present:  pip install xgboost
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg
from pipelines.preprocessing_pipeline import run as preprocess

# ------------------------------------------------------------
MODEL_PATH = os.path.join(cfg.MODELS_DIR, "xgboost.pkl")
# ------------------------------------------------------------


def _check_xgboost():
    try:
        import xgboost  # noqa: F401
    except ImportError:
        print("[ERROR] xgboost is not installed. Run:  pip install xgboost")
        sys.exit(1)


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
    _check_xgboost()
    from xgboost import XGBClassifier

    # -- Preprocessing check ---------------------------------------------------
    preprocess()

    # -- Model exists check ----------------------------------------------------
    if os.path.exists(MODEL_PATH):
        print(f"[OK] Model already exists at {MODEL_PATH}")
        print("   Skipping training. Delete the file to retrain.\n")
        ans = input("   Re-train anyway? [y/N]: ").strip().lower()
        if ans != "y":
            return

    # -- Load data ------------------------------------------------------------
    print("\n[1/3] Loading splits ...")
    train, val, test = load_splits()

    X_train = train[cfg.FEATURE_COLS].values
    y_train = train[cfg.TARGET_LABELS].values
    X_val   = val[cfg.FEATURE_COLS].values
    y_val   = val[cfg.TARGET_LABELS].values
    X_test  = test[cfg.FEATURE_COLS].values
    y_test  = test[cfg.TARGET_LABELS].values

    # -- Compute scale_pos_weight per label (handles class imbalance) ----------
    scale_weights = []
    for i, label in enumerate(cfg.TARGET_LABELS):
        pos = y_train[:, i].sum()
        neg = len(y_train) - pos
        scale_weights.append(neg / pos if pos > 0 else 1.0)

    # -- Train ------------------------------------------------------------
    print("[2/3] Training XGBoost (one model per label) ...")
    base_xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=cfg.RANDOM_STATE,
        verbosity=0,
    )
    model = MultiOutputClassifier(base_xgb, n_jobs=-1)
    model.fit(X_train, y_train)

    # -- Evaluate ------------------------------------------------------------
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

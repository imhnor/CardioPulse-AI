"""
evaluate_all.py
===============
Loads EVERY saved model (both ML and DL) and produces a unified
comparison table on the held-out TEST set.

What it prints:
  1. Per-model summary table (Hamming Loss, Macro F1, AUROC)
  2. Per-label F1 breakdown for each DL model
  3. Which model wins overall

Usage:
  python evaluate_all.py

Requirements:
  - Run preprocessing_pipeline first  (preprocessed/ must exist)
  - At least one model in models/     must have been trained
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

from sklearn.metrics import (
    classification_report, hamming_loss, f1_score, roc_auc_score
)

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
ML_MODELS = {
    "Logistic Regression" : os.path.join(cfg.MODELS_DIR, "logistic_regression.pkl"),
    "Random Forest"       : os.path.join(cfg.MODELS_DIR, "random_forest.pkl"),
    "XGBoost"             : os.path.join(cfg.MODELS_DIR, "xgboost.pkl"),
    "MLP (Tabular)"       : os.path.join(cfg.MODELS_DIR, "neural_network.pth"),
}

DL_MODELS = {
    "CNN-1D"       : os.path.join(cfg.MODELS_DIR, "cnn1d.pth"),
    "ResNet-1D"    : os.path.join(cfg.MODELS_DIR, "resnet1d.pth"),
    "CNN-BiLSTM"   : os.path.join(cfg.MODELS_DIR, "cnn_lstm.pth"),
}

SEPARATOR = "=" * 80


# -----------------------------------------------------------------------
# ML evaluation (tabular features)
# -----------------------------------------------------------------------

def eval_sklearn_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    hl     = hamming_loss(y_test, y_pred)
    mf1    = f1_score(y_test, y_pred, average="macro",  zero_division=0)
    uif1   = f1_score(y_test, y_pred, average="micro",  zero_division=0)
    per_cls= f1_score(y_test, y_pred, average=None,     zero_division=0)
    try:
        probs = np.array([e.predict_proba(X_test)[:, 1]
                          for e in model.estimators_]).T
        auroc = roc_auc_score(y_test, probs, average="macro")
    except Exception:
        auroc = float("nan")
    return {
        "hamming_loss" : hl,
        "macro_f1"     : mf1,
        "micro_f1"     : uif1,
        "per_class_f1" : per_cls.tolist(),
        "macro_auroc"  : auroc,
    }


# -----------------------------------------------------------------------
# DL evaluation (raw signal)
# -----------------------------------------------------------------------

def eval_dl_model(model_class, model_path, test_csv, device):
    from pipelines.utils.ecg_dataset import ECGSignalDataset
    from pipelines.utils.dl_trainer import evaluate_loader
    from torch.utils.data import DataLoader

    model = model_class(in_channels=12, num_classes=len(cfg.TARGET_LABELS))
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_ds = ECGSignalDataset(test_csv, augment=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    return evaluate_loader(model, test_dl, device)


def get_dl_classes():
    """Lazy import so that if torch not installed it won't crash ML eval."""
    from pipelines.train_cnn1d    import CNN1D
    from pipelines.train_resnet1d import ResNet1D
    from pipelines.train_cnn_lstm import CNNLSTM
    return {
        "CNN-1D"     : CNN1D,
        "ResNet-1D"  : ResNet1D,
        "CNN-BiLSTM" : CNNLSTM,
    }


# -----------------------------------------------------------------------
# Pretty printing
# -----------------------------------------------------------------------

def print_summary_table(results: dict):
    header = f"{'Model':<25} {'Hamming':>10} {'Macro F1':>10} {'Micro F1':>10} {'AUROC':>10}"
    print("\n" + SEPARATOR)
    print("  COMPARISON TABLE (Test Set)")
    print(SEPARATOR)
    print(header)
    print("-" * 70)

    best_f1 = max(v["macro_f1"] for v in results.values())
    for name, m in results.items():
        star = " <-- BEST" if abs(m["macro_f1"] - best_f1) < 1e-4 else ""
        auroc = f"{m['macro_auroc']:.4f}" if not np.isnan(m["macro_auroc"]) else "  N/A "
        print(f"  {name:<23} {m['hamming_loss']:>10.4f} {m['macro_f1']:>10.4f} "
              f"{m['micro_f1']:>10.4f} {auroc:>10}{star}")
    print(SEPARATOR)


def print_per_class(name: str, metrics: dict):
    print(f"\n  [{name}] Per-label F1 on Test Set:")
    for label, f1 in zip(cfg.TARGET_LABELS, metrics["per_class_f1"]):
        bar = "#" * int(f1 * 30)
        print(f"    {label:<5} : {f1:.4f}  {bar}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    # ---- Check preprocessing ----
    if not os.path.exists(cfg.TEST_CSV):
        print("[ERROR] Test split not found. Run preprocessing_pipeline.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # ---- ML models (tabular) ----
    test_df = pd.read_csv(cfg.TEST_CSV)
    X_test  = test_df[cfg.FEATURE_COLS].values
    y_test  = test_df[cfg.TARGET_LABELS].values

    print(f"\n{SEPARATOR}")
    print("  EVALUATING ML MODELS (tabular features: age/sex/height/weight/pacemaker)")
    print(SEPARATOR)

    for name, path in ML_MODELS.items():
        if name == "MLP (Tabular)":
            continue   # handled via separate torch logic below
        if not os.path.exists(path):
            print(f"  [SKIP] {name} -- model file not found ({path})")
            continue
        print(f"  Loading {name} ...", end=" ", flush=True)
        with open(path, "rb") as f:
            model = pickle.load(f)
        m = eval_sklearn_model(model, X_test, y_test)
        results[name] = m
        print(f"Macro F1={m['macro_f1']:.4f}  AUROC={m['macro_auroc']:.4f}")

    # ---- DL models (raw ECG signals) ----
    print(f"\n{SEPARATOR}")
    print("  EVALUATING DL MODELS (raw 12-lead ECG signals)")
    print(SEPARATOR)

    try:
        dl_classes = get_dl_classes()
    except ImportError as e:
        print(f"  [SKIP] DL models -- {e}")
        dl_classes = {}

    for name, path in DL_MODELS.items():
        if name not in dl_classes:
            continue
        if not os.path.exists(path):
            print(f"  [SKIP] {name} -- model file not found ({path})")
            continue
        print(f"  Loading {name} ...", end=" ", flush=True)
        try:
            m = eval_dl_model(dl_classes[name], path, cfg.TEST_CSV, device)
            results[name] = m
            print(f"Macro F1={m['macro_f1']:.4f}  AUROC={m['macro_auroc']:.4f}")
        except Exception as ex:
            print(f"  [ERROR] {name}: {ex}")

    if not results:
        print("\n[ERROR] No models could be evaluated. Train some models first.")
        return

    # ---- Summary table ----
    print_summary_table(results)

    # ---- Per-class breakdown for DL models ----
    for name in DL_MODELS.keys():
        if name in results:
            print_per_class(name, results[name])

    # ---- Best model ----
    best = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"\n  WINNER: {best}  (Macro F1 = {results[best]['macro_f1']:.4f})\n")


if __name__ == "__main__":
    main()

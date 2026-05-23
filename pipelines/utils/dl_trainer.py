"""
pipelines/utils/dl_trainer.py
------------------------------
Shared training loop used by all DL model pipelines.
Keeps each training script clean and DRY.

Features:
  - BCEWithLogitsLoss with pos_weight for class imbalance
  - AdamW optimizer + CosineAnnealingLR scheduler
  - Early stopping on validation loss
  - Per-epoch metrics logging (Macro F1, Hamming Loss)
  - Best model checkpoint auto-saved
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as cfg


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """
    Compute classification metrics from raw logits.

    Returns dict with:
        hamming_loss  : fraction of labels that are wrong
        macro_f1      : unweighted mean F1 across labels
        micro_f1      : globally averaged F1
        per_class_f1  : F1 for each superclass
        macro_auroc   : mean AUROC across labels (requires probs)
    """
    probs  = 1 / (1 + np.exp(-y_pred_logits))    # sigmoid
    preds  = (probs >= threshold).astype(int)

    hl   = hamming_loss(y_true, preds)
    mf1  = f1_score(y_true, preds, average="macro",  zero_division=0)
    uif1 = f1_score(y_true, preds, average="micro",  zero_division=0)

    per_cls = f1_score(y_true, preds, average=None, zero_division=0)

    try:
        auroc = roc_auc_score(y_true, probs, average="macro")
    except Exception:
        auroc = float("nan")

    return {
        "hamming_loss" : hl,
        "macro_f1"     : mf1,
        "micro_f1"     : uif1,
        "per_class_f1" : per_cls.tolist(),
        "macro_auroc"  : auroc,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model          : nn.Module,
    train_loader   : DataLoader,
    val_loader     : DataLoader,
    model_path     : str,
    epochs         : int   = 50,
    lr             : float = 1e-3,
    patience       : int   = 10,
    pos_weight_vec : torch.Tensor = None,
    device         : torch.device = None,
    weight_decay   : float = 1e-4,
) -> dict:
    """
    Train a PyTorch model and save the best checkpoint.

    Returns a dict with best val metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_vec.to(device) if pos_weight_vec is not None else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None
    best_metrics   = {}
    history        = {"train_loss": [], "val_loss": [], "val_macro_f1": []}

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)

        train_loss /= len(train_loader.dataset)

        # ---- Validate ----
        model.eval()
        val_loss   = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item() * len(Xb)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(yb.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        logits_np = np.concatenate(all_logits)
        labels_np = np.concatenate(all_labels)
        metrics   = compute_metrics(labels_np, logits_np)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(metrics["macro_f1"])

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  "
                  f"macro_F1={metrics['macro_f1']:.4f}  "
                  f"AUROC={metrics['macro_auroc']:.4f}")

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_metrics   = metrics
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # Restore best weights and save
    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n  Best checkpoint saved -> {model_path}")
    print(f"  Best val  Hamming Loss : {best_metrics['hamming_loss']:.4f}")
    print(f"  Best val  Macro F1     : {best_metrics['macro_f1']:.4f}")
    print(f"  Best val  Macro AUROC  : {best_metrics['macro_auroc']:.4f}")

    return best_metrics, history


# ---------------------------------------------------------------------------
# Evaluation on a DataLoader
# ---------------------------------------------------------------------------

def evaluate_loader(model: nn.Module, loader: DataLoader,
                    device: torch.device = None) -> dict:
    """Run model over a DataLoader and return metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval().to(device)
    all_logits, all_labels = [], []

    with torch.no_grad():
        for Xb, yb in loader:
            all_logits.append(model(Xb.to(device)).cpu().numpy())
            all_labels.append(yb.numpy())

    return compute_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_logits),
    )

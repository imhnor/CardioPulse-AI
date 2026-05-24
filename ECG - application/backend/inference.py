"""
inference.py
------------
Loads the trained ResNet1D model and runs predictions on preprocessed ECG tensors.

Model architecture MUST match the training definition in:
  pipelines/train_resnet1d.py

Input  : torch.Tensor of shape (1, 12, 1000)
Output : dict with sigmoid probabilities for each class
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

# ── Label definitions (must match training order) ─────────────────────────────
TARGET_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

LABEL_INFO = {
    "NORM": {"name": "Normal ECG",               "color": "#22c55e"},
    "MI":   {"name": "Myocardial Infarction",     "color": "#ef4444"},
    "STTC": {"name": "ST/T Change",               "color": "#f97316"},
    "CD":   {"name": "Conduction Disturbance",    "color": "#a855f7"},
    "HYP":  {"name": "Hypertrophy",               "color": "#3b82f6"},
}

# ── Threshold for binary prediction ───────────────────────────────────────────
THRESHOLD = 0.5

# ── Default model path (relative to this file → ../../models/) ────────────────
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "resnet1d.pth"
)


# ── Architecture ──────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """Standard 1-D Residual Block — identical to training code."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.relu     = nn.ReLU(inplace=True)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=pad, bias=False)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.drop     = nn.Dropout(0.2)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
        return self.relu(out + self.shortcut(x))


class ResNet1D(nn.Module):
    """
    4-stage 1-D ResNet for 12-lead ECG multi-label classification.
    Exact replica of the trained architecture.
    """
    def __init__(self, in_channels: int = 12, num_classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ResBlock1D(64, 64, kernel=7), ResBlock1D(64, 64, kernel=7))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128, stride=2, kernel=5), ResBlock1D(128, 128, kernel=5))
        self.layer3 = nn.Sequential(ResBlock1D(128, 256, stride=2, kernel=3), ResBlock1D(256, 256, kernel=3))
        self.layer4 = nn.Sequential(ResBlock1D(256, 256, stride=2, kernel=3), ResBlock1D(256, 256, kernel=3))
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(x)


# ── Singleton model loader ────────────────────────────────────────────────────

_model: ResNet1D | None = None
_device: torch.device | None = None


def load_model(model_path: str | None = None) -> ResNet1D:
    """Load model once and cache globally. Thread-safe for single-worker."""
    global _model, _device
    if _model is not None:
        return _model

    path = model_path or _DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model  = ResNet1D(in_channels=12, num_classes=len(TARGET_LABELS))

    state = torch.load(path, map_location=_device, weights_only=True)
    _model.load_state_dict(state)
    _model.to(_device)
    _model.eval()
    return _model


def predict(tensor: torch.Tensor) -> Dict:
    """
    Run inference on a preprocessed ECG tensor.

    Args:
        tensor: shape (1, 12, 1000) — float32

    Returns:
        dict with keys:
          probabilities : {label: float}  — sigmoid scores 0–1
          predictions   : {label: bool}   — threshold 0.5
          top_diagnosis : str             — highest-probability label
          is_normal     : bool            — NORM > threshold AND no pathology
          labels_info   : list of dicts for frontend rendering
    """
    model = load_model()

    with torch.no_grad():
        tensor = tensor.to(_device)
        logits = model(tensor)                          # (1, 5)
        probs  = torch.sigmoid(logits).cpu().numpy()[0] # (5,)

    probabilities = {label: float(round(float(p), 4)) for label, p in zip(TARGET_LABELS, probs)}
    predictions   = {label: bool(p >= THRESHOLD) for label, p in probabilities.items()}

    # Determine top diagnosis (highest probability)
    top_label = max(probabilities, key=probabilities.get)

    # Normal if NORM is highest AND no other class exceeds threshold
    pathology_active = any(predictions[l] for l in TARGET_LABELS if l != "NORM")
    is_normal = predictions["NORM"] and not pathology_active

    # Build rich label list for frontend
    labels_info = []
    for label in TARGET_LABELS:
        prob  = probabilities[label]
        info  = LABEL_INFO[label]
        labels_info.append({
            "code":       label,
            "name":       info["name"],
            "color":      info["color"],
            "probability": prob,
            "percent":    round(prob * 100, 1),
            "predicted":  predictions[label],
        })

    # Sort by probability descending for display
    labels_info.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "probabilities":  probabilities,
        "predictions":    predictions,
        "top_diagnosis":  LABEL_INFO[top_label]["name"],
        "top_code":       top_label,
        "top_confidence": probabilities[top_label],
        "is_normal":      is_normal,
        "labels_info":    labels_info,
    }

# ECG Multi-Label Classification — Complete Model Guide

> **Dataset:** PTB-XL (21,799 ECG records, 12 leads, 10 sec @ 100 Hz)  
> **Task:** Multi-label classification → predict which of 5 cardiac superclasses are present  
> **Labels:** `NORM` · `MI` · `STTC` · `CD` · `HYP`

---

## Table of Contents
1. [Dataset Understanding](#1-dataset-understanding)
2. [Why Multi-Label?](#2-why-multi-label)
3. [All Models — Why Each Was Chosen](#3-all-models--why-each-was-chosen)
4. [Metrics — What They Mean](#4-metrics--what-they-mean)
5. [How to Run Everything](#5-how-to-run-everything)
6. [How to Read Results](#6-how-to-read-results)
7. [Expected Performance](#7-expected-performance)
8. [Folder Structure](#8-folder-structure)

---

## 1. Dataset Understanding

| Property | Value |
|----------|-------|
| Records | 17,221 (after label filtering) |
| Train split | Folds 1-8 → **13,801** records |
| Val split | Fold 9 → **1,709** records |
| Test split | Fold 10 → **1,711** records |
| Signal shape | **(1000, 12)** — 1000 time-steps × 12 leads |
| Sampling rate | 100 Hz (10 seconds per recording) |
| Leads | I, II, III, AVR, AVL, AVF, V1–V6 |

### Class Distribution (approximate)
| Class | Description | Count | Notes |
|-------|-------------|-------|-------|
| NORM | Normal ECG | ~9,500 | Most common — class imbalance! |
| MI | Myocardial Infarction (heart attack) | ~5,500 | — |
| STTC | ST/T-Change | ~5,200 | — |
| CD | Conduction Disturbance | ~4,900 | — |
| HYP | Hypertrophy | ~2,600 | Least common |

> [!WARNING]
> **NORM dominates the dataset.** If you don't handle class imbalance, the model learns to always predict NORM and gets 55% accuracy while being medically useless.  
> **Solution used:** `pos_weight` in `BCEWithLogitsLoss` — penalises misses on minority classes.

---

## 2. Why Multi-Label?

A single ECG can show **multiple** conditions simultaneously.  
Example: A patient can have both `MI` AND `CD` at the same time.

```
Record #8:  {'IMI': 35.0, 'ABQRS': 0.0, 'SR': 0.0}
-> MI = 1, CD = 1   (two labels active at once)
```

This means we use **one sigmoid per output neuron** (not softmax),
and **Binary Cross-Entropy loss** — each label is an independent binary decision.

---

## 3. All Models — Why Each Was Chosen

### ML Models (tabular features: age, sex, height, weight, pacemaker)

---

#### Logistic Regression
**File:** `pipelines/train_logistic_regression.py`

| | |
|--|--|
| **Type** | Linear classifier |
| **Why use it** | Ultra-fast training (<1 min). Interpretable. Establishes a **minimum baseline** — any other model should beat this. |
| **Limitation** | Cannot learn non-linear patterns. Only 5 features. Expected weak performance. |
| **Class imbalance** | `class_weight="balanced"` |

---

#### Random Forest
**File:** `pipelines/train_random_forest.py`

| | |
|--|--|
| **Type** | Ensemble of 300 decision trees |
| **Why use it** | Handles non-linear interactions (e.g., *old + heavy + male = MI risk*). Robust to outliers. No feature scaling needed. |
| **Limitation** | Still only uses 5 demographic features — no signal information. |
| **Class imbalance** | `class_weight="balanced"` |

---

#### XGBoost
**File:** `pipelines/train_xgboost.py`

| | |
|--|--|
| **Type** | Gradient Boosted Trees |
| **Why use it** | Usually outperforms Random Forest on tabular data. Each new tree corrects errors of the previous. Better generalisation. |
| **Class imbalance** | `scale_pos_weight` per label |
| **Install** | `pip install xgboost` |

---

### DL Models (use raw 12-lead ECG waveform)

> [!IMPORTANT]
> These models read the **actual ECG signal** (all 1000 timesteps × 12 leads). This is where real performance improvement happens — demographics alone are insufficient for clinical ECG diagnosis.

---

#### CNN-1D (1D Convolutional Neural Network)
**File:** `pipelines/train_cnn1d.py`

```
Input (B, 12, 1000)
  -> Conv(32, k=7) -> BN -> ReLU -> MaxPool   [500]
  -> Conv(64, k=5) -> BN -> ReLU -> MaxPool   [250]
  -> Conv(128,k=5) -> BN -> ReLU -> MaxPool   [125]
  -> Conv(256,k=3) -> BN -> ReLU -> AvgPool   [1]
  -> Linear(256->5)
```

| | |
|--|--|
| **Why use it** | A 1D kernel sliding over time is the **most natural operation for ECG**. It detects wave shapes (P, QRS, T) at fixed scales — same as a cardiologist scanning waveforms. |
| **Strength** | Fast, good for morphological features |
| **Limitation** | No memory — cannot capture inter-beat patterns or RR interval trends |
| **Params** | ~200K |

**What each layer learns:**
- Layer 1 (k=7): Local peaks and slopes — individual wave components
- Layer 2 (k=5): Beat-level morphology — QRS complex shape  
- Layer 3 (k=5): Multi-beat patterns — repeated abnormalities
- Layer 4 (k=3): Abstract features — combined morphology

---

#### ResNet-1D (1D Residual Network)
**File:** `pipelines/train_resnet1d.py`

```
Input (B, 12, 1000)
  -> Stem Conv(64, k=15, stride=2)          [500]
  -> ResBlock(64->64)  x2                   [500]
  -> ResBlock(64->128, stride=2) x2         [250]
  -> ResBlock(128->256,stride=2) x2         [125]
  -> ResBlock(256->512,stride=2) x2         [63]
  -> AdaptiveAvgPool -> Linear(512->5)
```

Each ResBlock:
```
x -> [Conv -> BN -> ReLU -> Conv -> BN] + x  -> ReLU
                                         ^
                                    shortcut (skip connection)
```

| | |
|--|--|
| **Why use it** | Skip connections solve the **vanishing gradient problem** — allows training of 8+ conv layers. Published as **state-of-the-art for PTB-XL** (Strodthoff et al. 2021). Captures both fine morphology (shallow layers) and global rhythm (deep layers). |
| **Strength** | Best accuracy among CNN-based models. Most stable training. |
| **Limitation** | Slower than plain CNN |
| **Params** | ~1.8M |

> [!TIP]
> **Why depth matters:** Shallow layers see small local features (individual peaks). Deep layers see abstract combinations (full ST-segment in context of the entire beat cycle).

---

#### CNN-BiLSTM (Hybrid CNN + Bidirectional LSTM)
**File:** `pipelines/train_cnn_lstm.py`

```
Input (B, 12, 1000)
  -> CNN Feature Extractor -> (B, 128, 125)   [125 time-steps of 128-dim features]
  -> Transpose             -> (B, 125, 128)
  -> BiLSTM(hidden=128, layers=2)
     -- forward LSTM:  reads left->right
     -- backward LSTM: reads right->left      -> (B, 125, 256)
  -> Take last hidden state                   -> (B, 256)
  -> Dropout -> Linear(256->5)
```

| | |
|--|--|
| **Why use it** | ECG has **two types of patterns**: (1) local wave shapes (CNN), (2) temporal rhythm over 10 seconds (LSTM). Rhythm disorders (STTC, CD) need full temporal context. BiLSTM reads both **forward AND backward**, giving it full context over the recording. |
| **Strength** | Best for rhythm-based diagnoses (STTC, CD, atrial fibrillation) |
| **Limitation** | Slower training; LSTM is sequential |
| **Params** | ~1.2M |

---

### Model Comparison Summary

| Model | Uses Signal? | Morphology | Rhythm | Speed | Expected AUROC |
|-------|:-----------:|:----------:|:------:|-------|---------------|
| Logistic Regression | No | No | No | Fastest | ~0.65-0.70 |
| Random Forest | No | No | No | Fast | ~0.70-0.75 |
| XGBoost | No | No | No | Fast | ~0.72-0.77 |
| **CNN-1D** | **Yes** | **Yes** | No | Medium | **~0.85-0.90** |
| **ResNet-1D** | **Yes** | **Yes** | Partially | Medium | **~0.90-0.93** |
| **CNN-BiLSTM** | **Yes** | **Yes** | **Yes** | Slower | **~0.87-0.92** |

---

## 4. Metrics — What They Mean

### Hamming Loss
```
Hamming Loss = wrong label predictions / total label predictions
```
- Range: 0 (perfect) to 1 (all wrong)
- With 5 labels × 1711 test records = 8555 total decisions
- **Lower is better**
- `0.05` means 5% of all label decisions were wrong

### Macro F1 Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
Macro F1 = average F1 across all 5 classes (equal weight per class)
```
- NORM has many samples → easy to get high F1 on NORM alone
- Macro F1 gives **equal weight to rare classes (HYP) and common ones (NORM)**
- **Higher is better**
- `0.80` = average 80% F1 across all 5 diagnoses

### Micro F1 Score
```
Micro F1 = F1 computed on all predictions merged (weighted by frequency)
```
- Dominated by NORM (largest class)
- Good for understanding overall correctness
- Always higher than Macro F1

### AUROC (Area Under ROC Curve)
```
AUROC = P(model ranks a positive higher than a negative)
```
- Range: 0.5 (random) to 1.0 (perfect)
- **Threshold-independent** — measures overall discriminability
- **Higher is better**
- `0.90+` is considered clinically useful

### Per-Class F1
Shows per-disease performance:
```
NORM  : 0.91   <- easy, most data
MI    : 0.76   <- moderate
STTC  : 0.74   <- moderate
CD    : 0.72   <- harder
HYP   : 0.60   <- hardest (least data)
```

---

## 5. How to Run Everything

### One command for everything
```powershell
$env:PYTHONIOENCODING='utf-8'
.\ecg_env\Scripts\python.exe run_all.py
```

### Individual steps
```powershell
$env:PYTHONIOENCODING='utf-8'

# Preprocessing only
.\ecg_env\Scripts\python.exe pipelines/preprocessing_pipeline.py

# ML models
.\ecg_env\Scripts\python.exe pipelines/train_logistic_regression.py
.\ecg_env\Scripts\python.exe pipelines/train_random_forest.py
.\ecg_env\Scripts\python.exe pipelines/train_xgboost.py

# DL models (requires torch + wfdb)
.\ecg_env\Scripts\python.exe pipelines/train_cnn1d.py
.\ecg_env\Scripts\python.exe pipelines/train_resnet1d.py
.\ecg_env\Scripts\python.exe pipelines/train_cnn_lstm.py

# Compare all models
.\ecg_env\Scripts\python.exe evaluate_all.py
```

### Force re-train a model
```powershell
Remove-Item models/resnet1d.pth
.\ecg_env\Scripts\python.exe pipelines/train_resnet1d.py
```

### Force full re-preprocessing
```powershell
Remove-Item -Recurse preprocessed/
.\ecg_env\Scripts\python.exe pipelines/preprocessing_pipeline.py
```

---

## 6. How to Read Results

### During Training (printed every 5 epochs)
```
Epoch   5/50  train=0.4321  val=0.4105  macro_F1=0.6812  AUROC=0.8740
Epoch  10/50  train=0.3876  val=0.3642  macro_F1=0.7234  AUROC=0.8950
...
Early stopping at epoch 38.
Best val  Hamming Loss : 0.0812
Best val  Macro F1     : 0.7891
Best val  Macro AUROC  : 0.9124
```

**Good signs:**
- `val` loss going down each epoch
- `macro_F1` going up
- `train` and `val` loss close (no overfitting)

**Bad signs (problems):**

| Symptom | Problem | Fix |
|---------|---------|-----|
| `val` rises, `train` falls | Overfitting | Increase Dropout, reduce epochs |
| `macro_F1` stuck at ~0.2 | Model predicts all-negative | Check pos_weight |
| Loss = NaN | Exploding gradients | Reduce LR, check grad clipping |

### Final Comparison Table
```
================================================================================
  COMPARISON TABLE (Test Set)
================================================================================
  Model                     Hamming   Macro F1   Micro F1      AUROC
----------------------------------------------------------------------
  Logistic Regression        0.1523     0.5234     0.6123     0.6812
  Random Forest              0.1321     0.6012     0.6834     0.7234
  XGBoost                    0.1234     0.6234     0.6987     0.7456
  CNN-1D                     0.0923     0.7234     0.7812     0.8734
  ResNet-1D                  0.0812     0.7891     0.8234     0.9124  <-- BEST
  CNN-BiLSTM                 0.0876     0.7712     0.8102     0.8987
================================================================================
```

### Per-class breakdown
```
[ResNet-1D] Per-label F1 on Test Set:
  NORM  : 0.9123  ##############################
  MI    : 0.7623  ######################
  STTC  : 0.7412  ######################
  CD    : 0.7134  #####################
  HYP   : 0.6234  ##################
```

**Analysis:** HYP is always lowest because it has the fewest training samples.  
If you need better HYP performance → apply oversampling or stronger pos_weight.

---

## 7. Expected Performance

| Model | Macro AUROC | Notes |
|-------|------------|-------|
| Logistic Regression | 0.65-0.70 | Demographic features alone are weak |
| Random Forest | 0.70-0.75 | Non-linear combos help slightly |
| XGBoost | 0.72-0.77 | Best among tabular models |
| CNN-1D | 0.85-0.90 | Big jump — signal contains real diagnostic info |
| **ResNet-1D** | **0.90-0.93** | Best overall. Matches literature benchmark |
| CNN-BiLSTM | 0.87-0.92 | Better on rhythm disorders (STTC, CD) |

> [!NOTE]
> PTB-XL paper benchmark (Strodthoff et al. 2021):  
> ResNet variants → Macro AUROC **0.925** on test fold 10.  
> Our ResNet1D should reach ~0.90-0.92 given its architecture.

---

## 8. Folder Structure

```
ECG/
├── config.py                       <- All shared paths & hyperparameters
├── run_all.py                      <- Run entire pipeline in one command
├── evaluate_all.py                 <- Compare all models on test set
├── MODEL_GUIDE.md                  <- This file
│
├── pipelines/
│   ├── preprocessing_pipeline.py   <- Step 1: clean, split, scale, save
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_cnn1d.py              <- DL: local morphology
│   ├── train_resnet1d.py           <- DL: deep residual (recommended best)
│   ├── train_cnn_lstm.py           <- DL: morphology + temporal rhythm
│   └── utils/
│       ├── ecg_dataset.py          <- PyTorch Dataset for raw ECG signals
│       └── dl_trainer.py           <- Shared training loop & metrics
│
├── preprocessed/                   <- AUTO-GENERATED (delete to re-preprocess)
│   ├── ptbxl_cleaned.csv
│   ├── train.csv / val.csv / test.csv
│   └── scaler.pkl
│
└── models/                         <- AUTO-GENERATED (delete to re-train)
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── cnn1d.pth
    ├── resnet1d.pth
    └── cnn_lstm.pth
```

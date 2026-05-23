<div align="center">
  <h1>🫀 Automated Multi-Label ECG Classification</h1>
  <p><b>Advanced Deep Learning Framework for PTB-XL Clinical Dataset</b></p>
</div>

## 📖 Overview

This repository contains an end-to-end Machine Learning and Deep Learning pipeline designed to perform **multi-label classification** of 12-lead Electrocardiograms (ECG). Trained and validated on the **PTB-XL dataset**, the system predicts five major cardiac diagnostic superclasses simultaneously. 

We approach this problem with strict Data Science rigour. Our pipelines emphasize proper data handling to prevent leakage, sophisticated strategies to overcome extreme class imbalance, and strict regularization to prevent model overfitting in clinical scenarios.

---

## 🎯 Diagnostic Superclasses Predicted

1. **NORM**: Normal ECG
2. **MI**: Myocardial Infarction
3. **STTC**: ST/T Change
4. **CD**: Conduction Disturbance
5. **HYP**: Hypertrophy

---

## 🔬 Scientific Approach & Key Features

### 1. Robust Data Engineering
- **Imputation & Cleaning**: Demographic features (`age`, `height`, `weight`) are strictly imputed using training-set medians. String categories like `pacemaker` and `sex` are rigorously binary encoded.
- **Signal Processing**: The 100Hz 12-lead ECG `.dat` signals are loaded efficiently, standardized, and concatenated with normalized demographic inputs.
- **No Data Leakage**: We strictly adhere to the PTB-XL 10-fold split constraint (Fold 1-8 for training, Fold 9 for validation, Fold 10 for testing). Scalers are fit exclusively on the training fold.

### 2. State-of-the-Art Deep Learning Models
Instead of relying solely on classical ML with tabular data, our deep learning pipeline operates directly on the raw 1D signals:
- **1D-ResNet**: Deep residual architectures designed to map complex spatial relationships across all 12 leads without vanishing gradients.
- **CNN-BiLSTM**: Combines spatial feature extraction (CNN) with temporal sequence modelling (Bidirectional LSTM) to capture heart rhythms accurately over the 10-second window.

### 3. Handling Extreme Class Imbalance
To ensure minority classes (e.g., MI, HYP) are learned effectively:
- Implemented **Class Weights** to heavily penalize errors on underrepresented diagnostics.
- Supported **Focal Loss** to automatically focus learning on hard-to-predict samples.

### 4. Zero Tolerance for Overfitting
Models are guarded by clinical-grade regularization:
- Heavy `Dropout` and `Spatial Dropout` layers.
- `L2 Weight Decay` within AdamW optimizers.
- **Early Stopping** based strictly on validation Macro-F1 scores and Validation Loss.
- **1D Signal Augmentation** (Gaussian Noise, Baseline Wander) applied uniquely to the training set to encourage robust generalization.

---

## 📊 Evaluation & Clinical Validity

Because accuracy is heavily skewed by the majority class (`NORM`), our primary evaluation metrics are **AUROC** and **Macro-F1**. 
We provide comprehensive tools to generate:
- Classification Reports with precision/recall for all 5 classes.
- Optimal Decision Thresholds fine-tuned on the validation set.
- Detailed visual insights for feature importance and confusion matrices.

---

## 🛠️ Repository Structure

```
├── config.py                          # Global parameters, paths, and model configs
├── Deep_Learning_ECG_Detailed_Approach.md # In-depth technical methodology
├── pipelines/
│   ├── preprocessing_pipeline.py      # Cleans tabular metadata and saves splits
│   ├── utils/ecg_dataset.py           # Deep Learning PyTorch dataset class
│   └── train_dl_models.py             # Main PyTorch training scripts
├── models/                            # Saved `.pt` and `.pkl` artifacts
└── README.md                          # Project overview
```

---

## 🚀 How to Run

1. **Preprocess the Data**:
   ```bash
   python pipelines/preprocessing_pipeline.py
   ```
2. **Train Models**:
   ```bash
   python pipelines/train_resnet.py 
   ```
3. **Evaluate Results**:
   ```bash
   python evaluate_all.py
   ```

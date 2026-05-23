"""
preprocessing_pipeline.py
-------------------------
Run this ONCE (or whenever raw data changes).

What it does
  1. Loads  ptbxl_database.csv + scp_statements.csv
  2. Cleans / imputes demographics
  3. Encodes pacemaker column (string -> 0/1)
  4. Parses scp_codes dict -> multi-label binary columns (NORM/MI/STTC/CD/HYP)
  5. Removes records with NO confirmed diagnostic class
  6. Splits into train / val / test using PTB-XL stratified folds
  7. Scales numeric features and saves the fitted scaler
  8. Saves cleaned CSV, scaled splits and scaler to preprocessed/

Usage
  python pipelines/preprocessing_pipeline.py

Output files (all in preprocessed/)
  ptbxl_cleaned.csv  — full cleaned dataset (before scaling)
  train.csv          — training split  (folds 1-8)
  val.csv            — validation split (fold 9)
  test.csv           — held-out test   (fold 10)
  scaler.pkl         — fitted StandardScaler
"""

import os
import sys
import ast
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -- Allow running from repo root OR from pipelines/ ---------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _ensure_dirs():
    os.makedirs(cfg.PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(cfg.MODELS_DIR,       exist_ok=True)


def _load_raw():
    print("[1/6] Loading raw data ...")
    df  = pd.read_csv(cfg.RAW_CSV)
    scp = pd.read_csv(cfg.SCP_CSV, index_col=0)
    print(f"      Raw records : {len(df):,}")
    return df, scp


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Cleaning & imputing ...")

    columns_to_keep = [
        "ecg_id", "patient_id", "age", "sex", "height", "weight",
        "pacemaker", "strat_fold", "scp_codes", "filename_lr",
    ]
    df = df[columns_to_keep].copy()

    # -- numeric imputation ----------------------------------------------------
    df["age"]    = df["age"].fillna(df["age"].median())
    df["sex"]    = df["sex"].fillna(0)
    df["height"] = df["height"].fillna(df["height"].median())
    df["weight"] = df["weight"].fillna(df["weight"].median())

    # -- pacemaker: any non-null string (e.g. "1ES") -> 1, NaN -> 0 -------------
    df["pacemaker"] = df["pacemaker"].apply(
        lambda v: 0 if (pd.isna(v) or str(v).strip() in ("", "0", "False", "nan")) else 1
    )

    print(f"      Pacemaker positives : {df['pacemaker'].sum():,}")
    return df


def _encode_labels(df: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Encoding multi-label superclasses ...")

    scp_diag = scp[scp["diagnostic"] == 1.0]

    def extract_superclasses(codes_str):
        try:
            codes_dict = ast.literal_eval(codes_str)
        except Exception:
            return []
        classes = set()
        for code, score in codes_dict.items():
            if score == 100.0 and code in scp_diag.index:
                cls = scp_diag.loc[code, "diagnostic_class"]
                if pd.notna(cls):
                    classes.add(cls)
        return list(classes)

    df["superclasses"] = df["scp_codes"].apply(extract_superclasses)

    for sc in cfg.TARGET_LABELS:
        df[sc] = df["superclasses"].apply(lambda x: 1 if sc in x else 0)

    # Keep only records that have at least one confirmed class
    before = len(df)
    df["_total"] = df[cfg.TARGET_LABELS].sum(axis=1)
    df = df[df["_total"] > 0].copy()
    df.drop(columns=["_total", "superclasses", "scp_codes"], inplace=True)

    print(f"      Records after label filter : {len(df):,}  (dropped {before - len(df):,})")
    return df


def _split(df: pd.DataFrame):
    print("[4/6] Splitting train / val / test by strat_fold ...")
    test  = df[df["strat_fold"] == cfg.TEST_FOLD].copy()
    val   = df[df["strat_fold"] == cfg.VAL_FOLD].copy()
    train = df[~df["strat_fold"].isin([cfg.TEST_FOLD, cfg.VAL_FOLD])].copy()
    print(f"      Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")
    return train, val, test


def _scale(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    print("[5/6] Fitting scaler on train features ...")
    scaler = StandardScaler()
    train[cfg.FEATURE_COLS] = scaler.fit_transform(train[cfg.FEATURE_COLS])
    val[cfg.FEATURE_COLS]   = scaler.transform(val[cfg.FEATURE_COLS])
    test[cfg.FEATURE_COLS]  = scaler.transform(test[cfg.FEATURE_COLS])

    with open(cfg.SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    print(f"      Scaler saved -> {cfg.SCALER_FILE}")
    return train, val, test


def _save(df_full, train, val, test):
    print("[6/6] Saving artefacts ...")
    df_full.to_csv(cfg.CLEANED_CSV, index=False)
    train.to_csv(cfg.TRAIN_CSV, index=False)
    val.to_csv(cfg.VAL_CSV,   index=False)
    test.to_csv(cfg.TEST_CSV,  index=False)
    print(f"      Cleaned CSV -> {cfg.CLEANED_CSV}")
    print(f"      Train CSV   -> {cfg.TRAIN_CSV}")
    print(f"      Val CSV     -> {cfg.VAL_CSV}")
    print(f"      Test CSV    -> {cfg.TEST_CSV}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def run():
    _ensure_dirs()

    # Skip if already done
    all_exist = all(os.path.exists(p) for p in [
        cfg.CLEANED_CSV, cfg.TRAIN_CSV, cfg.VAL_CSV,
        cfg.TEST_CSV, cfg.SCALER_FILE,
    ])
    if all_exist:
        print("[OK] Preprocessed files already exist. Skipping preprocessing.")
        print("   Delete the 'preprocessed/' folder to force re-run.\n")
        return

    df, scp = _load_raw()
    df      = _clean(df)
    df      = _encode_labels(df, scp)
    train, val, test = _split(df)
    train, val, test = _scale(train, val, test)
    _save(df, train, val, test)

    print("\n[OK] Preprocessing complete!\n")


if __name__ == "__main__":
    run()

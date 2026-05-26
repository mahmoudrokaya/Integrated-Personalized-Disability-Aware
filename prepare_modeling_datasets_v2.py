#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Modeling-Ready NHANES Datasets
======================================

This version uses the confirmed folder structure:

D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/Pre-Processed-Cycles
    corrected/NHANES_pooled_filled_corrected.csv
    Data/NHANES_2011-2012_final_processed.csv
    Data/NHANES_2013-2014_final_processed.csv
    Data/NHANES_2015-2016_final_processed.csv
    Data/NHANES_2017-2018_final_processed.csv
    processed/NHANES_pooled_final_processed.csv

Later-cycle external file:
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/data_processed_later_cycles/corrected/NHANES_later_cycles_pooled_filled_corrected.csv

Outputs:
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/modeling_ready

Purpose:
1. Use 2011-2018 corrected pooled data for training/internal validation.
2. Use later-cycle corrected pooled data for external temporal validation.
3. Align shared predictors.
4. Create train/validation/test splits.
5. Save domain mapping for specialist teams.
6. Save reports for target distribution and feature alignment.

Run:
python prepare_modeling_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0) PATHS
# ============================================================

EXPERIMENTS_DIR = Path(
    r"D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments"
)

PREPROCESSED_DIR = EXPERIMENTS_DIR / "Pre-Processed-Cycles"

TRAINING_FILE = PREPROCESSED_DIR / "corrected" / "NHANES_pooled_filled_corrected.csv"

CYCLE_FILES = [
    PREPROCESSED_DIR / "Data" / "NHANES_2011-2012_final_processed.csv",
    PREPROCESSED_DIR / "Data" / "NHANES_2013-2014_final_processed.csv",
    PREPROCESSED_DIR / "Data" / "NHANES_2015-2016_final_processed.csv",
    PREPROCESSED_DIR / "Data" / "NHANES_2017-2018_final_processed.csv",
]

EXTERNAL_FILE = (
    EXPERIMENTS_DIR
    / "data_processed_later_cycles"
    / "corrected"
    / "NHANES_later_cycles_pooled_filled_corrected.csv"
)

OUT_DIR = EXPERIMENTS_DIR / "modeling_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) SETTINGS
# ============================================================

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_TEST_SPLIT = 0.50

TARGET_STAGE = "disability_stage"
TARGET_BINARY = "target_binary"

ADMIN_AND_TARGET_COLS = {
    "SEQN",
    "cycle",
    "SEQN_cycle_key",
    "target_source",
    "disability_stage",
    "target_binary",
    "upper_limb_difficulty",
    "mobility_difficulty",
    "adl_difficulty",
    "fallback_function_risk",
    "self_rated_health_risk",
}

DOMAIN_FEATURES = {
    "nutritional_metabolic": [
        "BMI",
        "height_cm",
        "weight_kg",
        "HbA1c",
        "fasting_glucose",
        "total_cholesterol",
        "HDL",
        "triglycerides",
        "LDL",
    ],
    "physiological": [
        "mean_SBP",
        "mean_DBP",
        "max_grip_strength",
    ],
    "behavioral": [
        "smoking_status",
        "MET_min_week",
        "meets_activity_guideline",
        "sedentary_minutes_day",
        "PHQ9_score",
        "PHQ9_category",
    ],
    "demographic_fairness": [
        "age",
        "sex",
        "race_ethnicity",
        "education",
        "income_poverty_ratio",
    ],
    "clinical_structural": [
        "arthritis",
        "arthritis_type",
        "medication_count",
        "polypharmacy",
    ],
}


# ============================================================
# 2) HELPERS
# ============================================================

def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:/n{path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def load_csv(path: Path) -> pd.DataFrame:
    return normalize_columns(pd.read_csv(path))


def make_internal_binary_target(stage: pd.Series) -> pd.Series:
    """
    Training/internal target:
      disability_stage 0 or 1 -> 0
      disability_stage 2 or 3 -> 1
    """
    s = pd.to_numeric(stage, errors="coerce")
    return np.where(
        s.isna(),
        np.nan,
        np.where(s >= 2, 1, np.where(s >= 0, 0, np.nan)),
    )


def make_external_binary_target(stage: pd.Series) -> pd.Series:
    """
    Later-cycle target:
    In the later-cycle pipeline, disability_stage stores WHQ_fallback_function_risk.
    It is already binary:
      0 -> lower functional risk
      1 -> higher functional risk
    """
    s = pd.to_numeric(stage, errors="coerce")
    return np.where(s.isna(), np.nan, np.where(s >= 1, 1, 0))


def get_feature_columns(train_df: pd.DataFrame, external_df: pd.DataFrame) -> List[str]:
    shared = sorted((set(train_df.columns) & set(external_df.columns)) - ADMIN_AND_TARGET_COLS)

    usable = []
    for col in shared:
        if train_df[col].notna().any() and external_df[col].notna().any():
            usable.append(col)

    return usable


def build_domain_map(features: List[str]) -> Dict[str, List[str]]:
    domain_map = {}
    used = set()

    for domain, candidates in DOMAIN_FEATURES.items():
        selected = [c for c in candidates if c in features]
        if selected:
            domain_map[domain] = selected
            used.update(selected)

    other = [c for c in features if c not in used]
    if other:
        domain_map["other_shared_predictors"] = other

    return domain_map


def convert_features_to_numeric(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def save_target_distribution(datasets: List[Tuple[str, pd.DataFrame]], out_path: Path) -> None:
    rows = []
    for name, df in datasets:
        counts = df[TARGET_BINARY].value_counts(dropna=False).sort_index()
        for cls, count in counts.items():
            rows.append({
                "dataset": name,
                "class": cls,
                "count": int(count),
                "percent": round((count / len(df)) * 100, 2) if len(df) else 0,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def save_alignment_report(train_df: pd.DataFrame, external_df: pd.DataFrame, features: List[str], out_path: Path) -> None:
    all_cols = sorted((set(train_df.columns) | set(external_df.columns)) - ADMIN_AND_TARGET_COLS)

    rows = []
    for col in all_cols:
        rows.append({
            "column": col,
            "in_2011_2018_training": col in train_df.columns,
            "in_later_external": col in external_df.columns,
            "selected_shared_predictor": col in features,
            "training_missing_percent": round(train_df[col].isna().mean() * 100, 2) if col in train_df.columns else None,
            "external_missing_percent": round(external_df[col].isna().mean() * 100, 2) if col in external_df.columns else None,
        })

    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def scale_using_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    external_df: pd.DataFrame,
    features: List[str],
):
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()
    ext_out = external_df.copy()

    train_medians = train_out[features].median(numeric_only=True)

    for frame in [train_out, val_out, test_out, ext_out]:
        frame[features] = frame[features].apply(pd.to_numeric, errors="coerce")
        frame[features] = frame[features].fillna(train_medians)

    scaler = StandardScaler()
    scaler.fit(train_out[features])

    train_out[features] = scaler.transform(train_out[features])
    val_out[features] = scaler.transform(val_out[features])
    test_out[features] = scaler.transform(test_out[features])
    ext_out[features] = scaler.transform(ext_out[features])

    scaler_info = {
        "feature_count": len(features),
        "features": features,
        "mean": {c: float(v) for c, v in zip(features, scaler.mean_)},
        "scale": {c: float(v) for c, v in zip(features, scaler.scale_)},
        "training_medians_used_for_remaining_missing": {
            c: (None if pd.isna(v) else float(v)) for c, v in train_medians.items()
        },
    }

    return train_out, val_out, test_out, ext_out, scaler_info


def save_cycle_overview() -> None:
    rows = []
    for path in CYCLE_FILES:
        if path.exists():
            df = load_csv(path)
            rows.append({
                "cycle_file": path.name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "has_disability_stage": TARGET_STAGE in df.columns,
                "disability_stage_nonmissing": int(df[TARGET_STAGE].notna().sum()) if TARGET_STAGE in df.columns else None,
            })
        else:
            rows.append({
                "cycle_file": path.name,
                "rows": None,
                "columns": None,
                "has_disability_stage": False,
                "disability_stage_nonmissing": None,
            })

    pd.DataFrame(rows).to_csv(
        OUT_DIR / "source_cycle_files_overview.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 3) MAIN
# ============================================================

def main() -> None:
    print("Preparing modeling-ready datasets")
    print(f"Preprocessed folder: {PREPROCESSED_DIR}")
    print(f"Training file: {TRAINING_FILE}")
    print(f"External file: {EXTERNAL_FILE}")
    print(f"Output folder: {OUT_DIR}")

    require_file(TRAINING_FILE, "2011-2018 corrected pooled training file")
    require_file(EXTERNAL_FILE, "Later-cycle corrected pooled external file")

    save_cycle_overview()

    train_source = load_csv(TRAINING_FILE)
    external_source = load_csv(EXTERNAL_FILE)

    if TARGET_STAGE not in train_source.columns:
        raise ValueError(f"Training file does not contain {TARGET_STAGE}")
    if TARGET_STAGE not in external_source.columns:
        raise ValueError(f"External file does not contain {TARGET_STAGE}")

    train_source[TARGET_BINARY] = make_internal_binary_target(train_source[TARGET_STAGE])
    external_source[TARGET_BINARY] = make_external_binary_target(external_source[TARGET_STAGE])

    train_source = train_source[train_source[TARGET_BINARY].notna()].copy()
    external_source = external_source[external_source[TARGET_BINARY].notna()].copy()

    train_source[TARGET_BINARY] = train_source[TARGET_BINARY].astype(int)
    external_source[TARGET_BINARY] = external_source[TARGET_BINARY].astype(int)

    features = get_feature_columns(train_source, external_source)

    if not features:
        raise ValueError("No usable shared predictors were found between training and later-cycle external data.")

    train_source = convert_features_to_numeric(train_source, features)
    external_source = convert_features_to_numeric(external_source, features)

    train_source = train_source[train_source[features].notna().any(axis=1)].copy()
    external_source = external_source[external_source[features].notna().any(axis=1)].copy()

    train_df, temp_df = train_test_split(
        train_source,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_source[TARGET_BINARY],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=VAL_TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_BINARY],
    )

    base_keep = ["SEQN", "cycle", TARGET_STAGE, TARGET_BINARY] + features
    train_df = train_df[[c for c in base_keep if c in train_df.columns]].copy()
    val_df = val_df[[c for c in base_keep if c in val_df.columns]].copy()
    test_df = test_df[[c for c in base_keep if c in test_df.columns]].copy()

    ext_keep = ["SEQN", "cycle", "target_source", TARGET_STAGE, TARGET_BINARY] + features
    external_source = external_source[[c for c in ext_keep if c in external_source.columns]].copy()

    train_df, val_df, test_df, external_source, scaler_info = scale_using_training(
        train_df,
        val_df,
        test_df,
        external_source,
        features,
    )

    train_df.to_csv(OUT_DIR / "train_2011_2018.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(OUT_DIR / "val_2011_2018.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(OUT_DIR / "test_2011_2018.csv", index=False, encoding="utf-8-sig")
    external_source.to_csv(OUT_DIR / "external_2017_2023.csv", index=False, encoding="utf-8-sig")

    domain_map = build_domain_map(features)

    with open(OUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    with open(OUT_DIR / "feature_domain_map.json", "w", encoding="utf-8") as f:
        json.dump(domain_map, f, indent=2)

    with open(OUT_DIR / "scaler_info.json", "w", encoding="utf-8") as f:
        json.dump(scaler_info, f, indent=2)

    save_target_distribution(
        [
            ("train_2011_2018", train_df),
            ("val_2011_2018", val_df),
            ("test_2011_2018", test_df),
            ("external_2017_2023", external_source),
        ],
        OUT_DIR / "target_distribution_report.csv",
    )

    save_alignment_report(
        train_source,
        external_source,
        features,
        OUT_DIR / "dataset_alignment_report.csv",
    )

    summary = {
        "training_file": str(TRAINING_FILE),
        "external_file": str(EXTERNAL_FILE),
        "output_folder": str(OUT_DIR),
        "selected_shared_feature_count": len(features),
        "selected_shared_features": features,
        "domain_map": domain_map,
        "rows": {
            "train_2011_2018": int(train_df.shape[0]),
            "val_2011_2018": int(val_df.shape[0]),
            "test_2011_2018": int(test_df.shape[0]),
            "external_2017_2023": int(external_source.shape[0]),
        },
        "columns": {
            "train_2011_2018": int(train_df.shape[1]),
            "val_2011_2018": int(val_df.shape[1]),
            "test_2011_2018": int(test_df.shape[1]),
            "external_2017_2023": int(external_source.shape[1]),
        },
        "target_notes": {
            "internal_2011_2018": "target_binary maps disability_stage 0-1 to 0 and 2-3 to 1.",
            "external_later_cycles": "target_binary uses later-cycle disability_stage generated from WHQ fallback functional risk.",
        },
        "scaling_notes": "Scaler fitted only on the 2011-2018 training split. Remaining missing values filled using training medians only.",
    }

    with open(OUT_DIR / "modeling_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUT_DIR / "modeling_dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write("Modeling-ready NHANES dataset preparation summary/n")
        f.write("=" * 60 + "/n/n")
        f.write(f"Training file: {TRAINING_FILE}/n")
        f.write(f"External file: {EXTERNAL_FILE}/n/n")
        f.write(f"Selected shared features: {len(features)}/n")
        for c in features:
            f.write(f"  - {c}/n")
        f.write("/nRows:/n")
        for k, v in summary["rows"].items():
            f.write(f"  {k}: {v}/n")
        f.write("/nDomain map:/n")
        for domain, cols in domain_map.items():
            f.write(f"  {domain}: {', '.join(cols)}/n")

    print("/nCompleted successfully.")
    print(f"Saved outputs to: {OUT_DIR}")
    print(f"Selected shared features: {len(features)}")
    print("Rows:")
    for k, v in summary["rows"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

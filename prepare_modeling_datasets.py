#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 0: Prepare Modeling-Ready NHANES Datasets
====================================================

Purpose
-------
Prepare the modeling-ready files required before Experiment 1.

This script aligns:
1) the main 2011-2018 corrected NHANES training dataset, and
2) the later-cycle corrected NHANES external validation dataset.

It creates:
- internal training, validation, and test splits from 2011-2018
- later-cycle external test file from 2017-2023
- feature-domain mapping for organizational specialist teams
- feature alignment report
- target distribution report

Design
------
Training/internal validation:
    target = disability_stage from 2011-2018

External temporal validation:
    target = disability_stage in later-cycle file
    note: this field represents WHQ_fallback_function_risk in the later-cycle pipeline

Feature strategy:
    Use only shared predictors that exist in both datasets and are not target/admin fields.

Author use
----------
Run from:
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/Codes

Command:
python prepare_modeling_datasets.py

Requirements:
pip install pandas numpy scikit-learn
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0) PATHS
# ============================================================

PROJECT_DIR = Path(r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments")

# Main 2011-2018 corrected dataset.
# The script searches several likely locations.
TRAINING_CANDIDATES = [
    Path(r"D:\47\472\New-Papers\Atlam1-2026\corrected\NHANES_pooled_filled_corrected.csv"),
    Path(r"D:\47\472\New-Papers\Atlam1-2026\NHANES_pooled_filled_corrected.csv"),
    PROJECT_DIR / "data_processed" / "corrected" / "NHANES_pooled_filled_corrected.csv",
    PROJECT_DIR / "data_processed" / "NHANES_pooled_filled_corrected.csv",
    PROJECT_DIR / "NHANES_pooled_filled_corrected.csv",
]

EXTERNAL_FILE = PROJECT_DIR / "data_processed_later_cycles" / "corrected" / "NHANES_later_cycles_pooled_filled_corrected.csv"

OUT_DIR = PROJECT_DIR / "modeling_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE_WITHIN_TEMP = 0.50  # temp 30% becomes 15% validation + 15% internal test

TARGET = "disability_stage"

ADMIN_AND_TARGET_COLS = {
    "SEQN",
    "cycle",
    "SEQN_cycle_key",
    "target_source",
    "disability_stage",
    "upper_limb_difficulty",
    "mobility_difficulty",
    "adl_difficulty",
    "fallback_function_risk",
    "self_rated_health_risk",
}

# Domain mapping must reflect Section 3 and Table 1.
DOMAIN_FEATURES = {
    "nutritional_metabolic": [
        "BMI",
        "HbA1c",
        "fasting_glucose",
        "total_cholesterol",
        "HDL",
        "triglycerides",
        "LDL",
        "height_cm",
        "weight_kg",
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

def find_existing_training_file() -> Path:
    for p in TRAINING_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find the 2011-2018 corrected training dataset. Checked:\n"
        + "\n".join(str(p) for p in TRAINING_CANDIDATES)
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def make_binary_target_from_stage(series: pd.Series) -> pd.Series:
    """
    Main binary target:
      stage 0 or 1 -> 0
      stage 2 or 3 -> 1

    For later-cycle data, disability_stage already represents WHQ fallback binary risk
    in the current pipeline. This function still preserves 0/1 values.
    """
    s = pd.to_numeric(series, errors="coerce")
    return np.where(s.isna(), np.nan, np.where(s >= 2, 1, np.where(s >= 0, s, np.nan)))


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def target_distribution(df: pd.DataFrame, target_col: str, dataset_name: str) -> pd.DataFrame:
    vc = df[target_col].value_counts(dropna=False).sort_index()
    rows = []
    for cls, count in vc.items():
        rows.append({
            "dataset": dataset_name,
            "target": target_col,
            "class": cls,
            "count": int(count),
            "percent": round(float(count) / len(df) * 100, 2) if len(df) else 0,
        })
    return pd.DataFrame(rows)


def choose_shared_features(train_df: pd.DataFrame, external_df: pd.DataFrame) -> List[str]:
    train_cols = set(train_df.columns)
    ext_cols = set(external_df.columns)

    shared = sorted((train_cols & ext_cols) - ADMIN_AND_TARGET_COLS)

    # Keep columns with at least one non-missing value in both datasets.
    usable = []
    for c in shared:
        if train_df[c].notna().any() and external_df[c].notna().any():
            usable.append(c)

    return usable


def build_domain_map(feature_cols: List[str]) -> Dict[str, List[str]]:
    domain_map = {}
    used = set()

    for domain, candidates in DOMAIN_FEATURES.items():
        cols = [c for c in candidates if c in feature_cols]
        domain_map[domain] = cols
        used.update(cols)

    other = [c for c in feature_cols if c not in used]
    if other:
        domain_map["other_shared_predictors"] = other

    # Remove empty domains but keep the domain mapping report separate.
    return {k: v for k, v in domain_map.items() if v}


def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, external_df: pd.DataFrame, features: List[str]):
    scaler = StandardScaler()

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()
    external_scaled = external_df.copy()

    # All features should already be imputed, but fill any remaining external gaps conservatively
    # using training medians to avoid leakage from later cycles into training.
    medians = train_scaled[features].median(numeric_only=True)
    for frame in [train_scaled, val_scaled, test_scaled, external_scaled]:
        frame[features] = frame[features].apply(pd.to_numeric, errors="coerce")
        frame[features] = frame[features].fillna(medians)

    scaler.fit(train_scaled[features])

    train_scaled[features] = scaler.transform(train_scaled[features])
    val_scaled[features] = scaler.transform(val_scaled[features])
    test_scaled[features] = scaler.transform(test_scaled[features])
    external_scaled[features] = scaler.transform(external_scaled[features])

    scaler_info = {
        "features": features,
        "mean": {c: float(v) for c, v in zip(features, scaler.mean_)},
        "scale": {c: float(v) for c, v in zip(features, scaler.scale_)},
        "training_medians_used_for_remaining_missing": {c: (None if pd.isna(v) else float(v)) for c, v in medians.items()},
    }

    return train_scaled, val_scaled, test_scaled, external_scaled, scaler_info


# ============================================================
# 3) MAIN
# ============================================================

def main() -> None:
    print("Preparing modeling-ready NHANES datasets")

    training_file = find_existing_training_file()
    if not EXTERNAL_FILE.exists():
        raise FileNotFoundError(f"External later-cycle file not found: {EXTERNAL_FILE}")

    print(f"Training/internal file: {training_file}")
    print(f"External later-cycle file: {EXTERNAL_FILE}")
    print(f"Output folder: {OUT_DIR}")

    train_source = normalize_columns(pd.read_csv(training_file))
    external_source = normalize_columns(pd.read_csv(EXTERNAL_FILE))

    if TARGET not in train_source.columns:
        raise ValueError(f"Training file does not contain target column: {TARGET}")
    if TARGET not in external_source.columns:
        raise ValueError(f"External file does not contain target column: {TARGET}")

    # Create explicit binary target for modeling.
    train_source["target_binary"] = make_binary_target_from_stage(train_source[TARGET])
    external_source["target_binary"] = pd.to_numeric(external_source[TARGET], errors="coerce")

    # Keep valid target rows only.
    train_source = train_source[train_source["target_binary"].notna()].copy()
    external_source = external_source[external_source["target_binary"].notna()].copy()

    train_source["target_binary"] = train_source["target_binary"].astype(int)
    external_source["target_binary"] = external_source["target_binary"].astype(int)

    feature_cols = choose_shared_features(train_source, external_source)
    if not feature_cols:
        raise ValueError("No shared usable predictors found between training and external datasets.")

    train_source = safe_numeric(train_source, feature_cols)
    external_source = safe_numeric(external_source, feature_cols)

    # Drop rows still missing all features.
    train_source = train_source[train_source[feature_cols].notna().any(axis=1)].copy()
    external_source = external_source[external_source[feature_cols].notna().any(axis=1)].copy()

    # Stratified internal split.
    train_df, temp_df = train_test_split(
        train_source,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_source["target_binary"],
    )

    val_df, internal_test_df = train_test_split(
        temp_df,
        test_size=VAL_SIZE_WITHIN_TEMP,
        random_state=RANDOM_STATE,
        stratify=temp_df["target_binary"],
    )

    # Keep a compact modeling schema.
    keep_cols = ["SEQN", "cycle", "target_binary", TARGET] + feature_cols
    keep_cols = [c for c in keep_cols if c in train_source.columns]

    ext_keep_cols = ["SEQN", "cycle", "target_source", "target_binary", TARGET] + feature_cols
    ext_keep_cols = [c for c in ext_keep_cols if c in external_source.columns]

    train_model = train_df[keep_cols].copy()
    val_model = val_df[keep_cols].copy()
    internal_test_model = internal_test_df[keep_cols].copy()
    external_model = external_source[ext_keep_cols].copy()

    # Scale feature columns based on training split only.
    train_model, val_model, internal_test_model, external_model, scaler_info = scale_features(
        train_model,
        val_model,
        internal_test_model,
        external_model,
        feature_cols,
    )

    # Save datasets.
    train_path = OUT_DIR / "train_2011_2018.csv"
    val_path = OUT_DIR / "val_2011_2018.csv"
    test_path = OUT_DIR / "test_2011_2018.csv"
    external_path = OUT_DIR / "external_2017_2023.csv"

    train_model.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_model.to_csv(val_path, index=False, encoding="utf-8-sig")
    internal_test_model.to_csv(test_path, index=False, encoding="utf-8-sig")
    external_model.to_csv(external_path, index=False, encoding="utf-8-sig")

    # Save feature map.
    domain_map = build_domain_map(feature_cols)

    with open(OUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    with open(OUT_DIR / "feature_domain_map.json", "w", encoding="utf-8") as f:
        json.dump(domain_map, f, indent=2)

    with open(OUT_DIR / "scaler_info.json", "w", encoding="utf-8") as f:
        json.dump(scaler_info, f, indent=2)

    # Reports.
    distributions = pd.concat([
        target_distribution(train_model, "target_binary", "train_2011_2018"),
        target_distribution(val_model, "target_binary", "val_2011_2018"),
        target_distribution(internal_test_model, "target_binary", "test_2011_2018"),
        target_distribution(external_model, "target_binary", "external_2017_2023"),
    ], ignore_index=True)

    distributions.to_csv(OUT_DIR / "target_distribution_report.csv", index=False, encoding="utf-8-sig")

    alignment_rows = []
    all_candidate_cols = sorted((set(train_source.columns) | set(external_source.columns)) - ADMIN_AND_TARGET_COLS - {"target_binary"})
    for c in all_candidate_cols:
        alignment_rows.append({
            "column": c,
            "in_training_2011_2018": c in train_source.columns,
            "in_external_2017_2023": c in external_source.columns,
            "selected_as_shared_feature": c in feature_cols,
            "training_missing_percent": round(train_source[c].isna().mean() * 100, 2) if c in train_source.columns else None,
            "external_missing_percent": round(external_source[c].isna().mean() * 100, 2) if c in external_source.columns else None,
        })

    alignment_df = pd.DataFrame(alignment_rows)
    alignment_df.to_csv(OUT_DIR / "dataset_alignment_report.csv", index=False, encoding="utf-8-sig")

    summary = {
        "training_file": str(training_file),
        "external_file": str(EXTERNAL_FILE),
        "output_folder": str(OUT_DIR),
        "selected_shared_feature_count": len(feature_cols),
        "selected_shared_features": feature_cols,
        "domain_map": domain_map,
        "rows": {
            "train_2011_2018": int(train_model.shape[0]),
            "val_2011_2018": int(val_model.shape[0]),
            "test_2011_2018": int(internal_test_model.shape[0]),
            "external_2017_2023": int(external_model.shape[0]),
        },
        "columns": {
            "train_2011_2018": int(train_model.shape[1]),
            "val_2011_2018": int(val_model.shape[1]),
            "test_2011_2018": int(internal_test_model.shape[1]),
            "external_2017_2023": int(external_model.shape[1]),
        },
        "notes": [
            "Internal target_binary maps disability_stage 0-1 to 0 and 2-3 to 1.",
            "External target_binary uses later-cycle disability_stage, which represents WHQ fallback functional risk in the later-cycle pipeline.",
            "Feature scaling was fitted only on the 2011-2018 training split.",
            "Remaining missing values were filled using training medians only to avoid leakage from later-cycle data."
        ]
    }

    with open(OUT_DIR / "modeling_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUT_DIR / "modeling_dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write("Modeling-ready NHANES dataset preparation summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training/internal file: {training_file}\n")
        f.write(f"External file: {EXTERNAL_FILE}\n\n")
        f.write(f"Selected shared feature count: {len(feature_cols)}\n")
        f.write("Selected shared features:\n")
        for c in feature_cols:
            f.write(f"  - {c}\n")
        f.write("\nRows:\n")
        for k, v in summary["rows"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nDomain map:\n")
        for domain, cols in domain_map.items():
            f.write(f"  {domain}: {', '.join(cols)}\n")

    print("\nCompleted successfully.")
    print(f"Saved modeling-ready datasets to: {OUT_DIR}")
    print(f"Shared features: {len(feature_cols)}")
    print("Rows:")
    for k, v in summary["rows"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

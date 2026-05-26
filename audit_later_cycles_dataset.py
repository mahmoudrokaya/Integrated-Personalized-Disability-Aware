#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit script for pooled later-cycle NHANES corrected dataset
===========================================================

Input:
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/data_processed_later_cycles/corrected/NHANES_later_cycles_pooled_filled_corrected.csv

Outputs:
audit/
    dataset_shape_summary.csv
    target_distribution.csv
    missingness_summary.csv
    numerical_summary.csv
    logical_validation_report.txt
    logical_validation_report.json
    correlation_matrix.csv
    audit_summary.txt
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

INPUT_FILE = Path(
    r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\data_processed_later_cycles\corrected\NHANES_later_cycles_pooled_filled_corrected.csv"
)

OUTPUT_DIR = INPUT_FILE.parent / "audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")

# ============================================================
# 1) SHAPE SUMMARY
# ============================================================

shape_df = pd.DataFrame({
    "rows": [df.shape[0]],
    "columns": [df.shape[1]],
    "duplicate_rows": [int(df.duplicated().sum())],
    "duplicate_SEQN": [int(df['SEQN'].duplicated().sum()) if 'SEQN' in df.columns else -1],
})

shape_path = OUTPUT_DIR / "dataset_shape_summary.csv"
shape_df.to_csv(shape_path, index=False)

# ============================================================
# 2) TARGET DISTRIBUTION
# ============================================================

target_rows = []

for target in [
    "disability_stage",
    "fallback_function_risk",
    "self_rated_health_risk",
]:
    if target in df.columns:
        vc = df[target].value_counts(dropna=False).sort_index()
        for k, v in vc.items():
            target_rows.append({
                "target": target,
                "class": k,
                "count": int(v),
                "percent": round((v / len(df)) * 100, 2),
            })

target_df = pd.DataFrame(target_rows)

target_path = OUTPUT_DIR / "target_distribution.csv"
target_df.to_csv(target_path, index=False)

# ============================================================
# 3) MISSINGNESS
# ============================================================

missing_rows = []

for col in df.columns:
    miss = int(df[col].isna().sum())
    pct = round((miss / len(df)) * 100, 2)

    if pct == 0:
        degree = "None"
    elif pct < 10:
        degree = "Low"
    elif pct < 50:
        degree = "Moderate"
    else:
        degree = "Very High"

    missing_rows.append({
        "column": col,
        "missing_count": miss,
        "missing_percent": pct,
        "degree": degree,
    })

missing_df = pd.DataFrame(missing_rows).sort_values(
    "missing_percent",
    ascending=False
)

missing_path = OUTPUT_DIR / "missingness_summary.csv"
missing_df.to_csv(missing_path, index=False)

# ============================================================
# 4) NUMERICAL SUMMARY
# ============================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

summary_df = df[numeric_cols].describe().T

summary_df["median"] = df[numeric_cols].median()
summary_df["missing_percent"] = (
    df[numeric_cols].isna().mean() * 100
).round(2)

summary_path = OUTPUT_DIR / "numerical_summary.csv"
summary_df.to_csv(summary_path)

# ============================================================
# 5) LOGICAL VALIDATION
# ============================================================

checks = {}

# BMI validation
if {"BMI", "weight_kg", "height_cm"}.issubset(df.columns):
    bmi_calc = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
    bmi_error = (df["BMI"] - bmi_calc).abs()

    checks["BMI_error_gt_2"] = int((bmi_error > 2).sum())
    checks["BMI_negative"] = int((df["BMI"] < 0).sum())
    checks["BMI_gt_100"] = int((df["BMI"] > 100).sum())

# LDL validation
if {"LDL", "total_cholesterol", "HDL", "triglycerides"}.issubset(df.columns):
    ldl_calc = (
        df["total_cholesterol"]
        - df["HDL"]
        - (df["triglycerides"] / 5)
    )

    ldl_error = (df["LDL"] - ldl_calc).abs()

    checks["LDL_error_gt_20"] = int((ldl_error > 20).sum())
    checks["LDL_gt_total_cholesterol"] = int(
        (df["LDL"] > df["total_cholesterol"]).sum()
    )

# Blood pressure validation
if {"mean_SBP", "mean_DBP"}.issubset(df.columns):
    checks["DBP_ge_SBP"] = int(
        (df["mean_DBP"] >= df["mean_SBP"]).sum()
    )

# PHQ9 validation
if "PHQ9_score" in df.columns:
    checks["PHQ9_invalid_negative"] = int(
        (df["PHQ9_score"] < 0).sum()
    )

    checks["PHQ9_gt_27"] = int(
        (df["PHQ9_score"] > 27).sum()
    )

# MET validation
if {"MET_min_week", "meets_activity_guideline"}.issubset(df.columns):
    inconsistent = (
        ((df["MET_min_week"] >= 600) &
         (df["meets_activity_guideline"] != 1))
        |
        ((df["MET_min_week"] < 600) &
         (df["meets_activity_guideline"] != 0))
    )

    checks["MET_guideline_inconsistency"] = int(inconsistent.sum())

# Duplicate check
checks["duplicate_rows"] = int(df.duplicated().sum())

if "SEQN" in df.columns:
    checks["duplicate_SEQN"] = int(df["SEQN"].duplicated().sum())

json_path = OUTPUT_DIR / "logical_validation_report.json"
json_path.write_text(
    json.dumps(checks, indent=2),
    encoding="utf-8"
)

txt_path = OUTPUT_DIR / "logical_validation_report.txt"

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("Later-Cycle NHANES Audit Report\n")
    f.write("=" * 45 + "\n\n")

    for k, v in checks.items():
        f.write(f"{k}: {v}\n")

# ============================================================
# 6) CORRELATION MATRIX
# ============================================================

corr_cols = [
    c for c in [
        "BMI",
        "HbA1c",
        "fasting_glucose",
        "mean_SBP",
        "mean_DBP",
        "PHQ9_score",
        "MET_min_week",
        "age",
        "LDL",
    ]
    if c in df.columns
]

if len(corr_cols) >= 2:
    corr_df = df[corr_cols].corr(numeric_only=True)

    corr_path = OUTPUT_DIR / "correlation_matrix.csv"
    corr_df.to_csv(corr_path)

# ============================================================
# 7) FINAL TEXT SUMMARY
# ============================================================

summary_txt = OUTPUT_DIR / "audit_summary.txt"

with open(summary_txt, "w", encoding="utf-8") as f:

    f.write("Later-Cycle NHANES Audit Summary\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Dataset rows: {df.shape[0]}\n")
    f.write(f"Dataset columns: {df.shape[1]}\n\n")

    if not target_df.empty:
        f.write("Target distributions:\n")
        f.write(target_df.to_string(index=False))
        f.write("\n\n")

    f.write("Top missing variables:\n")
    f.write(
        missing_df.head(15).to_string(index=False)
    )
    f.write("\n\n")

    f.write("Logical validation:\n")
    for k, v in checks.items():
        f.write(f"{k}: {v}\n")

print("\nAudit completed successfully.")
print(f"Audit folder: {OUTPUT_DIR}")

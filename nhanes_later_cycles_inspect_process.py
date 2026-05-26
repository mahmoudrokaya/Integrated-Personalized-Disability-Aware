# -*- coding: utf-8 -*-
"""
NHANES Later-Cycle Folder Inspector and Processor
=================================================

Purpose
-------
Inspect the downloaded NHANES raw XPT folder structure and produce processed
cycle-level CSV files compatible with the earlier 2011-2018 pipeline.

Input folder
------------
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/data_raw/Downloaded

Output folder
-------------
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/data_processed_later_cycles

What this script does
---------------------
1. Recursively inspects the downloaded folder.
2. Builds an inventory of all XPT files.
3. Infers NHANES release/cycle labels from folder names and file names.
4. Finds required NHANES files for each cycle/release.
5. Processes each available cycle into a harmonized CSV.
6. Saves validation reports and missing-file reports.

Requirements
------------
pip install pandas numpy pyreadstat openpyxl

Notes
-----
NHANES later releases are not always standard yearly cycles.
Expected examples:
- 2017-March 2020 Pre-Pandemic: files often use P_ prefix, e.g., P_DEMO.xpt
- 2021-August 2023: files may use suffix L, e.g., DEMO_L.xpt
- 2025-2026: files may use newer suffixes when released

The script is intentionally tolerant. If optional files are missing, it fills the
corresponding variables with NaN. If core files are missing, it skips that cycle
and writes a clear report.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 0) USER PATHS
# ============================================================

RAW_DIR = Path(r"D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/data_raw/Downloaded")
OUT_DIR = Path(r"D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/data_processed_later_cycles")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) PROJECT FILE CODES
# ============================================================

# Standard NHANES components needed for the disability-health paper.
# Some later cycles may not contain every file.
PROJECT_CODES = {
    "DEMO": ["DEMO"],
    "BMX": ["BMX"],
    "BPX": ["BPX", "BPXO"],
    "MGX": ["MGX"],          # optional
    "GHB": ["GHB"],
    "GLU": ["GLU"],
    "TCHOL": ["TCHOL"],
    "HDL": ["HDL"],
    "TRIGLY": ["TRIGLY", "TRIGLY_H", "TRIGLY_I"],
    "PFQ": ["PFQ", "PFQ_H", "PFQ_I"],
    "SMQ": ["SMQ"],
    "PAQ": ["PAQ"],
    "DPQ": ["DPQ"],
    "MCQ": ["MCQ"],
    "RXQ_RX": ["RXQ_RX", "RXQ_DRUG", "RXQ"],
}

CORE_KEYS = ["DEMO", "BMX", "BPX", "PFQ"]
STRONGLY_RECOMMENDED_KEYS = ["GHB", "GLU", "TCHOL", "HDL", "TRIGLY", "SMQ", "PAQ", "DPQ", "MCQ", "RXQ_RX"]
OPTIONAL_KEYS = ["MGX"]

# Labels that may appear in downloaded folder paths.
CYCLE_PATTERNS = [
    (r"2017.*2020|pre.*pandemic|prepandemic", "2017-March_2020_PrePandemic"),
    (r"2021.*2023|august.*2023|2021-2023", "2021-August_2023"),
    (r"2025.*2026|2025-2026", "2025-2026"),
]

# File name hints for release/cycle if folder inference fails.
FILE_CYCLE_HINTS = [
    (r"^P_", "2017-March_2020_PrePandemic"),
    (r"_L$", "2021-August_2023"),
    (r"_M$", "2025-2026"),
]


# ============================================================
# 2) GENERAL HELPERS
# ============================================================

def read_xpt(path: Path) -> pd.DataFrame:
    """Read NHANES XPT file with encoding fallback."""
    try:
        return pd.read_sas(path, format="xport", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_sas(path, format="xport", encoding="latin1")


def keep_existing(df: pd.DataFrame, cols: List[Optional[str]]) -> pd.DataFrame:
    cols = [c for c in cols if c and c in df.columns]
    return df[cols].copy()


def replace_special_missing(series: pd.Series) -> pd.Series:
    return series.replace({
        5: np.nan, 7: np.nan, 9: np.nan,
        77: np.nan, 99: np.nan,
        777: np.nan, 999: np.nan,
        7777: np.nan, 9999: np.nan,
        77777: np.nan, 99999: np.nan,
    })


def numeric_clean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = replace_special_missing(df[c])
    return df


def mean_with_min_valid(row: pd.Series, cols: List[str], min_valid: int = 2) -> float:
    if not cols:
        return np.nan
    vals = pd.to_numeric(row[cols], errors="coerce")
    vals = vals.replace(0, np.nan)
    if vals.notna().sum() < min_valid:
        return np.nan
    return vals.mean()


def sum_with_all_required(row: pd.Series, cols: List[str]) -> float:
    existing = [c for c in cols if c in row.index]
    if len(existing) != len(cols):
        return np.nan
    vals = pd.to_numeric(row[existing], errors="coerce")
    if vals.notna().sum() != len(cols):
        return np.nan
    return vals.sum()


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fix_tiny_values_to_zero(series: pd.Series, threshold: float = 1e-10) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    return series.mask(series.abs() < threshold, 0)


def normalize_file_code(path: Path) -> str:
    """
    Convert filename to normalized NHANES code.
    Examples:
      P_DEMO.xpt  -> DEMO
      DEMO_L.xpt  -> DEMO
      RXQ_RX_L.xpt -> RXQ_RX
    """
    stem = path.stem.upper()
    stem = re.sub(r"^P_", "", stem)
    stem = re.sub(r"_[A-Z]$", "", stem)
    return stem


def infer_cycle_from_path(path: Path) -> str:
    path_text = str(path).lower()

    for pattern, label in CYCLE_PATTERNS:
        if re.search(pattern, path_text, flags=re.IGNORECASE):
            return label

    stem = path.stem.upper()
    for pattern, label in FILE_CYCLE_HINTS:
        if re.search(pattern, stem):
            return label

    return "Unknown_Cycle"


def candidate_priority(path: Path) -> Tuple[int, int]:
    """
    Prefer exact project files and shorter names.
    Lower tuple is better.
    """
    name = path.name.upper()
    priority = 0
    if name.startswith("P_"):
        priority -= 1
    if re.search(r"_[A-Z]\.XPT$", name):
        priority -= 1
    return (priority, len(name))


# ============================================================
# 3) INSPECT FOLDER STRUCTURE
# ============================================================

def build_inventory(raw_dir: Path) -> pd.DataFrame:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {raw_dir}")

    xpt_files = sorted(raw_dir.rglob("*.xpt")) + sorted(raw_dir.rglob("*.XPT"))

    rows = []
    for path in xpt_files:
        rows.append({
            "cycle_inferred": infer_cycle_from_path(path),
            "component_folder": path.parent.name,
            "normalized_code": normalize_file_code(path),
            "filename": path.name,
            "relative_path": str(path.relative_to(raw_dir)),
            "full_path": str(path),
            "file_size_bytes": path.stat().st_size,
        })

    inventory = pd.DataFrame(rows)

    if inventory.empty:
        print(f"No XPT files found under: {raw_dir}")
    else:
        inventory = inventory.drop_duplicates(subset=["full_path"]).reset_index(drop=True)

    inventory_path = OUT_DIR / "later_cycles_xpt_inventory.csv"
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    print(f"Saved inventory: {inventory_path}")

    return inventory


def print_folder_summary(inventory: pd.DataFrame) -> None:
    print("\n================ FOLDER INVENTORY SUMMARY ================\n")
    if inventory.empty:
        return

    summary = (
        inventory.groupby(["cycle_inferred", "component_folder"], as_index=False)
        .agg(
            files=("filename", "count"),
            size_mb=("file_size_bytes", lambda x: round(x.sum() / (1024 * 1024), 2)),
        )
        .sort_values(["cycle_inferred", "component_folder"])
    )
    print(summary.to_string(index=False))

    code_summary = (
        inventory.groupby(["cycle_inferred", "normalized_code"], as_index=False)
        .agg(files=("filename", "count"))
        .sort_values(["cycle_inferred", "normalized_code"])
    )
    code_summary_path = OUT_DIR / "later_cycles_code_summary.csv"
    code_summary.to_csv(code_summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved code summary: {code_summary_path}")


def find_cycle_files(inventory: pd.DataFrame, cycle_label: str) -> Dict[str, Path]:
    cycle_df = inventory[inventory["cycle_inferred"] == cycle_label].copy()
    files_map = {}

    for key, possible_codes in PROJECT_CODES.items():
        matches = cycle_df[cycle_df["normalized_code"].isin([c.upper() for c in possible_codes])].copy()

        if matches.empty:
            continue

        paths = [Path(p) for p in matches["full_path"].tolist()]
        paths = sorted(paths, key=candidate_priority)
        files_map[key] = paths[0]

    return files_map


# ============================================================
# 4) PROCESS ONE CYCLE
# ============================================================

def empty_by_seqn(base: pd.DataFrame, col: str) -> pd.DataFrame:
    out = base[["SEQN"]].copy()
    out[col] = np.nan
    return out


def prepare_demo(path: Path) -> pd.DataFrame:
    df = read_xpt(path)
    df = keep_existing(df, ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3", "DMDEDUC2", "INDFMPIR"])
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    df["age"] = df.get("RIDAGEYR", np.nan)
    df["sex"] = df.get("RIAGENDR", np.nan)

    # Use RIDRETH1 when available for compatibility with earlier processed files.
    if "RIDRETH1" in df.columns:
        df["race_ethnicity"] = df["RIDRETH1"]
    elif "RIDRETH3" in df.columns:
        df["race_ethnicity"] = df["RIDRETH3"]
    else:
        df["race_ethnicity"] = np.nan

    df["education"] = df.get("DMDEDUC2", np.nan)
    df["income_poverty_ratio"] = df.get("INDFMPIR", np.nan)
    df["income_poverty_ratio"] = fix_tiny_values_to_zero(df["income_poverty_ratio"])
    return df[["SEQN", "age", "sex", "race_ethnicity", "education", "income_poverty_ratio"]]


def prepare_bmx(path: Path, demo: pd.DataFrame) -> pd.DataFrame:
    if path is None:
        out = demo[["SEQN"]].copy()
        for c in ["BMI", "height_cm", "weight_kg"]:
            out[c] = np.nan
        return out

    df = read_xpt(path)
    df = keep_existing(df, ["SEQN", "BMXBMI", "BMXHT", "BMXWT"])
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    df["BMI"] = df.get("BMXBMI", np.nan)
    df["height_cm"] = df.get("BMXHT", np.nan)
    df["weight_kg"] = df.get("BMXWT", np.nan)

    df.loc[(df["BMI"] < 10) | (df["BMI"] > 80), "BMI"] = np.nan
    df.loc[(df["height_cm"] < 100) | (df["height_cm"] > 230), "height_cm"] = np.nan
    df.loc[(df["weight_kg"] < 20) | (df["weight_kg"] > 400), "weight_kg"] = np.nan
    return df[["SEQN", "BMI", "height_cm", "weight_kg"]]


def prepare_bpx(path: Path, demo: pd.DataFrame) -> pd.DataFrame:
    if path is None:
        out = demo[["SEQN"]].copy()
        out["mean_SBP"] = np.nan
        out["mean_DBP"] = np.nan
        return out

    df = read_xpt(path)
    df = keep_existing(df, [
        "SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
        "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"
    ])
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    sbp_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in df.columns]
    dbp_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in df.columns]

    df["mean_SBP"] = df.apply(lambda r: mean_with_min_valid(r, sbp_cols, min_valid=2), axis=1)
    df["mean_DBP"] = df.apply(lambda r: mean_with_min_valid(r, dbp_cols, min_valid=2), axis=1)

    df.loc[(df["mean_SBP"] < 70) | (df["mean_SBP"] > 250), "mean_SBP"] = np.nan
    df.loc[(df["mean_DBP"] < 40) | (df["mean_DBP"] > 150), "mean_DBP"] = np.nan
    return df[["SEQN", "mean_SBP", "mean_DBP"]]


def prepare_lab(path: Optional[Path], demo: pd.DataFrame, raw_col_candidates: List[str], out_col: str,
                low: Optional[float] = None, high: Optional[float] = None) -> pd.DataFrame:
    if path is None:
        return empty_by_seqn(demo, out_col)

    df = read_xpt(path)
    raw_cols = ["SEQN"] + raw_col_candidates
    df = keep_existing(df, raw_cols)
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    raw_col = first_existing(df, raw_col_candidates)
    df[out_col] = df[raw_col] if raw_col else np.nan

    if low is not None:
        df.loc[df[out_col] < low, out_col] = np.nan
    if high is not None:
        df.loc[df[out_col] > high, out_col] = np.nan

    return df[["SEQN", out_col]]


def prepare_mgx(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    if path is None:
        return empty_by_seqn(demo, "max_grip_strength")

    df = read_xpt(path)
    grip_cols = [
        "MGXH1T1", "MGXH1T2", "MGXH1T3",
        "MGXH2T1", "MGXH2T2", "MGXH2T3"
    ]
    df = keep_existing(df, ["SEQN"] + grip_cols)
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    existing = [c for c in grip_cols if c in df.columns]
    if not existing:
        df["max_grip_strength"] = np.nan
    else:
        for c in existing:
            df.loc[(df[c] < 1) | (df[c] > 100), c] = np.nan
        df["max_grip_strength"] = df[existing].max(axis=1, skipna=True)

    return df[["SEQN", "max_grip_strength"]]


def prepare_pfq(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["upper_limb_difficulty", "mobility_difficulty", "adl_difficulty", "disability_stage"]
    if path is None:
        out = demo[["SEQN"]].copy()
        for c in out_cols:
            out[c] = np.nan
        return out

    df = read_xpt(path)

    # Original 2011-2018 columns.
    pfq_items_old = ["PFQ061P", "PFQ061E", "PFQ061B", "PFQ061C", "PFQ061M", "PFQ061K", "PFQ061L"]

    # Keep flexible, because later releases may include new/changed PFQ columns.
    possible = ["SEQN"] + pfq_items_old + [c for c in df.columns if c.startswith("PFQ")]
    df = keep_existing(df, list(dict.fromkeys(possible)))
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    def difficulty_binary(x):
        if pd.isna(x):
            return np.nan
        if x == 1:
            return 0
        if x in [2, 3, 4]:
            return 1
        return np.nan

    existing_old = [c for c in pfq_items_old if c in df.columns]

    if len(existing_old) >= 3:
        for c in existing_old:
            df[c + "_bin"] = df[c].apply(difficulty_binary)

        def max_existing(cols):
            cols = [c for c in cols if c in df.columns]
            if not cols:
                return np.nan
            return df[cols].max(axis=1, skipna=True)

        df["upper_limb_difficulty"] = max_existing(["PFQ061P_bin", "PFQ061E_bin"])
        df["mobility_difficulty"] = max_existing(["PFQ061B_bin", "PFQ061C_bin", "PFQ061M_bin"])
        df["adl_difficulty"] = max_existing(["PFQ061K_bin", "PFQ061L_bin"])

        def assign_stage(row):
            vals = [row[c + "_bin"] for c in existing_old if c + "_bin" in row.index]
            if all(pd.isna(v) for v in vals):
                return np.nan
            if row.get("adl_difficulty", np.nan) == 1:
                return 3
            if row.get("mobility_difficulty", np.nan) == 1:
                return 2
            if row.get("upper_limb_difficulty", np.nan) == 1:
                return 1
            observed = [v for v in vals if not pd.isna(v)]
            if len(observed) > 0 and max(observed) == 0:
                return 0
            return np.nan

        df["disability_stage"] = df.apply(assign_stage, axis=1)
    else:
        # If PFQ structure changed, keep stage missing and report later.
        for c in out_cols:
            df[c] = np.nan

    return df[["SEQN"] + out_cols]


def prepare_smq(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    if path is None:
        return empty_by_seqn(demo, "smoking_status")

    df = read_xpt(path)
    df = keep_existing(df, ["SEQN", "SMQ020", "SMQ040"])
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    def smoking_status(row):
        ever = row.get("SMQ020", np.nan)
        now = row.get("SMQ040", np.nan)
        if pd.isna(ever):
            return np.nan
        if ever == 2:
            return 0
        if ever == 1:
            if now in [1, 2]:
                return 2
            if now == 3:
                return 1
            return 1
        return np.nan

    df["smoking_status"] = df.apply(smoking_status, axis=1)
    return df[["SEQN", "smoking_status"]]


def prepare_paq(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    out_cols = ["MET_min_week", "meets_activity_guideline", "sedentary_minutes_day"]
    if path is None:
        out = demo[["SEQN"]].copy()
        for c in out_cols:
            out[c] = np.nan
        return out

    df = read_xpt(path)
    cols = [
        "SEQN",
        "PAQ605", "PAQ610", "PAD615",
        "PAQ620", "PAQ625", "PAD630",
        "PAQ635", "PAQ640", "PAD645",
        "PAQ650", "PAQ655", "PAD660",
        "PAQ665", "PAQ670", "PAD675",
        "PAD680"
    ]
    df = keep_existing(df, cols)
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    def activity_component(gate, days, mins, met_value):
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        if gate not in df.columns:
            return out
        gate_yes = df[gate] == 1
        gate_no = df[gate] == 2
        if days in df.columns and mins in df.columns:
            idx = gate_yes & df[days].notna() & df[mins].notna()
            out.loc[idx] = df.loc[idx, days] * df.loc[idx, mins] * met_value
        out.loc[gate_no] = 0
        return out

    df["vig_work_met"] = activity_component("PAQ605", "PAQ610", "PAD615", 8)
    df["mod_work_met"] = activity_component("PAQ620", "PAQ625", "PAD630", 4)
    df["transport_met"] = activity_component("PAQ635", "PAQ640", "PAD645", 4)
    df["vig_rec_met"] = activity_component("PAQ650", "PAQ655", "PAD660", 8)
    df["mod_rec_met"] = activity_component("PAQ665", "PAQ670", "PAD675", 4)

    comps = ["vig_work_met", "mod_work_met", "transport_met", "vig_rec_met", "mod_rec_met"]
    df["MET_min_week"] = df[comps].sum(axis=1, min_count=1)
    df["MET_min_week"] = fix_tiny_values_to_zero(df["MET_min_week"])

    df["meets_activity_guideline"] = np.where(
        df["MET_min_week"].isna(),
        np.nan,
        np.where(df["MET_min_week"] >= 600, 1, 0)
    )

    if "PAD680" in df.columns:
        df["sedentary_minutes_day"] = df["PAD680"]
        df.loc[(df["sedentary_minutes_day"] < 0) | (df["sedentary_minutes_day"] > 1440), "sedentary_minutes_day"] = np.nan
    else:
        df["sedentary_minutes_day"] = np.nan

    return df[["SEQN"] + out_cols]


def prepare_dpq(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    if path is None:
        out = demo[["SEQN"]].copy()
        out["PHQ9_score"] = np.nan
        out["PHQ9_category"] = np.nan
        return out

    df = read_xpt(path)
    items = ["DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090"]
    df = keep_existing(df, ["SEQN"] + items)
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    def phq_category(score):
        if pd.isna(score):
            return np.nan
        if score < 5:
            return 0
        if score < 10:
            return 1
        if score < 15:
            return 2
        if score < 20:
            return 3
        return 4

    if all(c in df.columns for c in items):
        df["PHQ9_score"] = df.apply(lambda row: sum_with_all_required(row, items), axis=1)
    else:
        df["PHQ9_score"] = np.nan

    df["PHQ9_score"] = fix_tiny_values_to_zero(df["PHQ9_score"])
    df["PHQ9_category"] = df["PHQ9_score"].apply(phq_category)
    return df[["SEQN", "PHQ9_score", "PHQ9_category"]]


def prepare_mcq(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    if path is None:
        out = demo[["SEQN"]].copy()
        out["arthritis"] = np.nan
        out["arthritis_type"] = np.nan
        return out

    df = read_xpt(path)
    df = keep_existing(df, ["SEQN", "MCQ160A", "MCQ195"])
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    df["arthritis"] = np.where(
        df.get("MCQ160A", np.nan).isna(), np.nan,
        np.where(df["MCQ160A"] == 1, 1, np.where(df["MCQ160A"] == 2, 0, np.nan))
    )
    df["arthritis_type"] = df["MCQ195"] if "MCQ195" in df.columns else np.nan
    return df[["SEQN", "arthritis", "arthritis_type"]]


def prepare_rxq(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    out = demo[["SEQN"]].copy()
    out["medication_count"] = np.nan
    out["polypharmacy"] = np.nan

    if path is None:
        return out

    df = read_xpt(path)
    # RXDDRUG is older pipeline variable. Later files may use RXDRSC1 etc, but drug names still often exist.
    drug_col = first_existing(df, ["RXDDRUG", "RXDDRGID", "RXDRSC1", "RXDRSC2", "RXDRSC3"])
    df = keep_existing(df, ["SEQN", drug_col])
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")

    if drug_col is None:
        return out

    df = df[df[drug_col].notna()].copy()
    if df.empty:
        return out

    rx_agg = df.groupby("SEQN", as_index=False)[drug_col].nunique()
    rx_agg.rename(columns={drug_col: "medication_count"}, inplace=True)
    rx_agg["polypharmacy"] = np.where(rx_agg["medication_count"] >= 5, 1, 0)

    out = out.drop(columns=["medication_count", "polypharmacy"]).merge(rx_agg, on="SEQN", how="left")
    return out


def process_cycle(cycle_label: str, files_map: Dict[str, Path]) -> Optional[pd.DataFrame]:
    print(f"\n================ PROCESSING {cycle_label} ================\n")

    missing_core = [k for k in CORE_KEYS if k not in files_map]
    missing_recommended = [k for k in STRONGLY_RECOMMENDED_KEYS if k not in files_map]
    missing_optional = [k for k in OPTIONAL_KEYS if k not in files_map]

    report = {
        "cycle_label": cycle_label,
        "missing_core": missing_core,
        "missing_recommended": missing_recommended,
        "missing_optional": missing_optional,
        "files_used": {k: str(v) for k, v in files_map.items()},
    }

    report_path = OUT_DIR / f"{cycle_label}_file_availability_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if missing_core:
        print(f"SKIPPED {cycle_label}. Missing core files: {missing_core}")
        return None

    print("Files used:")
    for k, v in files_map.items():
        print(f"  {k}: {v.name}")

    if missing_recommended:
        print(f"Warning: missing recommended files will be filled with NaN: {missing_recommended}")
    if missing_optional:
        print(f"Optional missing files: {missing_optional}")

    demo = prepare_demo(files_map["DEMO"])
    bmx = prepare_bmx(files_map.get("BMX"), demo)
    bpx = prepare_bpx(files_map.get("BPX"), demo)
    mgx = prepare_mgx(files_map.get("MGX"), demo)

    ghb = prepare_lab(files_map.get("GHB"), demo, ["LBXGH"], "HbA1c", 3, 20)
    glu = prepare_lab(files_map.get("GLU"), demo, ["LBXGLU"], "fasting_glucose", 20, 800)
    tchol = prepare_lab(files_map.get("TCHOL"), demo, ["LBXTC"], "total_cholesterol", 50, 500)
    hdl = prepare_lab(files_map.get("HDL"), demo, ["LBDHDD", "LBXHDD"], "HDL", 10, 200)
    trig = prepare_lab(files_map.get("TRIGLY"), demo, ["LBXTR"], "triglycerides", 20, 3000)

    pfq = prepare_pfq(files_map.get("PFQ"), demo)
    smq = prepare_smq(files_map.get("SMQ"), demo)
    paq = prepare_paq(files_map.get("PAQ"), demo)
    dpq = prepare_dpq(files_map.get("DPQ"), demo)
    mcq = prepare_mcq(files_map.get("MCQ"), demo)
    rxq = prepare_rxq(files_map.get("RXQ_RX"), demo)

    dfs = [demo, bmx, bpx, mgx, ghb, glu, tchol, hdl, trig, pfq, smq, paq, dpq, mcq, rxq]

    final_df = dfs[0]
    for d in dfs[1:]:
        final_df = final_df.merge(d, on="SEQN", how="left")

    final_df["LDL"] = np.where(
        final_df["triglycerides"].notna()
        & final_df["total_cholesterol"].notna()
        & final_df["HDL"].notna()
        & (final_df["triglycerides"] < 400),
        final_df["total_cholesterol"] - final_df["HDL"] - (final_df["triglycerides"] / 5.0),
        np.nan,
    )

    for col in ["PHQ9_score", "income_poverty_ratio", "MET_min_week", "mean_SBP", "mean_DBP", "BMI"]:
        if col in final_df.columns:
            final_df[col] = fix_tiny_values_to_zero(final_df[col])

    # Match earlier 2011-2018 processing.
    final_df = final_df[final_df["age"] >= 20].copy()

    # Keep all rows even if disability_stage missing, but save a filtered version too.
    final_df["cycle"] = cycle_label

    preferred_cols = [
        "SEQN", "cycle",
        "disability_stage", "upper_limb_difficulty", "mobility_difficulty", "adl_difficulty",
        "age", "sex", "race_ethnicity", "education", "income_poverty_ratio",
        "BMI", "height_cm", "weight_kg", "mean_SBP", "mean_DBP", "max_grip_strength",
        "HbA1c", "fasting_glucose", "total_cholesterol", "HDL", "triglycerides", "LDL",
        "smoking_status", "MET_min_week", "meets_activity_guideline", "sedentary_minutes_day",
        "PHQ9_score", "PHQ9_category", "arthritis", "arthritis_type",
        "medication_count", "polypharmacy",
    ]
    final_df = final_df[[c for c in preferred_cols if c in final_df.columns]]

    # Save unfiltered and disability-stage-filtered versions.
    out_all = OUT_DIR / f"NHANES_{cycle_label}_processed_all_adults.csv"
    final_df.to_csv(out_all, index=False, encoding="utf-8-sig")

    filtered_df = final_df[final_df["disability_stage"].notna()].copy()
    out_filtered = OUT_DIR / f"NHANES_{cycle_label}_final_processed.csv"
    filtered_df.to_csv(out_filtered, index=False, encoding="utf-8-sig")

    validation = {
        "cycle_label": cycle_label,
        "all_adults_shape": list(final_df.shape),
        "filtered_shape_disability_stage_notna": list(filtered_df.shape),
        "disability_stage_counts": filtered_df["disability_stage"].value_counts(dropna=False).sort_index().to_dict()
            if "disability_stage" in filtered_df.columns else {},
        "missingness_percent_all_adults": (final_df.isna().mean() * 100).round(1).sort_values(ascending=False).to_dict(),
        "saved_all_adults": str(out_all),
        "saved_filtered": str(out_filtered),
    }

    validation_path = OUT_DIR / f"{cycle_label}_validation_report.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    print(f"Saved all-adults processed file: {out_all}")
    print(f"Saved final disability-stage processed file: {out_filtered}")
    print(f"Final filtered shape: {filtered_df.shape}")

    return filtered_df


# ============================================================
# 5) MAIN
# ============================================================

def main():
    print("NHANES later-cycle inspector and processor")
    print(f"Raw input folder: {RAW_DIR}")
    print(f"Output folder: {OUT_DIR}")

    inventory = build_inventory(RAW_DIR)
    print_folder_summary(inventory)

    if inventory.empty:
        print("No data to process.")
        return

    available_cycles = sorted(inventory["cycle_inferred"].dropna().unique().tolist())
    print("\nDetected cycles/releases:")
    for c in available_cycles:
        print(f"  - {c}")

    processed = []
    process_log = []

    for cycle_label in available_cycles:
        if cycle_label == "Unknown_Cycle":
            print("\nSkipping Unknown_Cycle. Inspect inventory manually.")
            continue

        files_map = find_cycle_files(inventory, cycle_label)
        df = process_cycle(cycle_label, files_map)

        process_log.append({
            "cycle_label": cycle_label,
            "processed": df is not None,
            "rows": int(df.shape[0]) if df is not None else 0,
            "columns": int(df.shape[1]) if df is not None else 0,
            "available_keys": ", ".join(sorted(files_map.keys())),
            "missing_core": ", ".join([k for k in CORE_KEYS if k not in files_map]),
        })

        if df is not None and not df.empty:
            processed.append(df)

    process_log_df = pd.DataFrame(process_log)
    process_log_path = OUT_DIR / "later_cycles_processing_log.csv"
    process_log_df.to_csv(process_log_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved processing log: {process_log_path}")

    if processed:
        pooled = pd.concat(processed, ignore_index=True)
        pooled_path = OUT_DIR / "NHANES_later_cycles_pooled_final_processed.csv"
        pooled.to_csv(pooled_path, index=False, encoding="utf-8-sig")
        print(f"Saved pooled later-cycle file: {pooled_path}")
        print(f"Pooled shape: {pooled.shape}")

        print("\nPooled disability stage counts:")
        print(pooled["disability_stage"].value_counts(dropna=False).sort_index())

    print("\nDone.")
    print("Inspect these files next:")
    print(f"  - {OUT_DIR / 'later_cycles_xpt_inventory.csv'}")
    print(f"  - {OUT_DIR / 'later_cycles_processing_log.csv'}")
    print(f"  - cycle-specific validation_report.json files")


if __name__ == "__main__":
    main()

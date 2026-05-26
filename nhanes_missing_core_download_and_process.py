# -*- coding: utf-8 -*-
"""
NHANES Missing Core Downloader + Later-Cycle Processor
======================================================

Why this version?
-----------------
Your previous run found later-cycle files but skipped processing because the
Demographics and PFQ/physical-function files were missing from the downloaded
folder. This script does three things:

1. Inspects the current folder structure.
2. Downloads missing DEMO files directly from CDC pages.
3. Processes available later cycles using the best available disability/function
   source:
      - PFQ when available
      - WHQ as fallback for activity limitation if PFQ is not available
      - HSQ as fallback for self-rated health if no disability-function file exists

Important:
----------
NHANES later releases do not always contain the same PFQ variables used in
2011-2018. Therefore, the script saves:
- all-adults processed files
- final files filtered by available disability/function target
- a target-source report explaining which outcome was used

Requirements:
-------------
pip install requests beautifulsoup4 pandas numpy pyreadstat tqdm openpyxl
"""

from __future__ import annotations

import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, unquote

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ============================================================
# 0) PATHS
# ============================================================

RAW_DIR = Path(r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\data_raw\Downloaded")
OUT_DIR = Path(r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\data_processed_later_cycles")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) CDC SETTINGS
# ============================================================

SEARCH_PAGE = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx"

TARGET_CYCLES = {
    "2017-March_2020_PrePandemic": "2017-2020",
    "2021-August_2023": "2021-2023",
}

COMPONENTS_TO_CHECK = ["Demographics", "Questionnaire", "Examination", "Laboratory", "Dietary"]

# Required or useful file codes. Later-cycle filenames may be P_DEMO or DEMO_L.
PROJECT_CODES = {
    "DEMO": ["DEMO"],
    "BMX": ["BMX"],
    "BPX": ["BPX", "BPXO"],
    "MGX": ["MGX"],
    "GHB": ["GHB"],
    "GLU": ["GLU"],
    "TCHOL": ["TCHOL"],
    "HDL": ["HDL"],
    "TRIGLY": ["TRIGLY"],
    "PFQ": ["PFQ"],
    "WHQ": ["WHQ"],   # fallback, includes limitation/weight-history fields in some releases
    "HSQ": ["HSQ"],   # fallback, current health status
    "SMQ": ["SMQ"],
    "PAQ": ["PAQ"],
    "DPQ": ["DPQ"],
    "MCQ": ["MCQ"],
    "RXQ_RX": ["RXQ_RX", "RXQ_DRUG", "RXQ"],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120 Safari/537.36"
}
session = requests.Session()
session.headers.update(HEADERS)


# ============================================================
# 2) DOWNLOAD HELPERS
# ============================================================

def safe_filename(name: str) -> str:
    name = unquote(name)
    name = re.sub(r"[<>:\"/\\|?*]+", "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name


def get_html(url: str, timeout: int = 30) -> str:
    last_err = None
    for attempt in range(3):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_err = exc
            time.sleep(2 + attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def discover_xpt_links(cycle_label: str, cycle_code: str, component: str) -> List[Dict]:
    url = f"{SEARCH_PAGE}?Component={component}&Cycle={cycle_code}"
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")

    records = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        visible = " ".join(a.get_text(" ", strip=True).split())

        if ".xpt" not in href.lower() and "data [xpt" not in visible.lower():
            continue

        full_url = urljoin(url, href)
        filename = safe_filename(Path(urlparse(full_url).path).name)

        if not filename.lower().endswith(".xpt"):
            filename = safe_filename(Path(href).name)

        if not filename.lower().endswith(".xpt"):
            continue

        tr = a.find_parent("tr")
        description = " ".join(tr.get_text(" ", strip=True).split()) if tr else visible

        records.append({
            "cycle_label": cycle_label,
            "cycle_code": cycle_code,
            "component": component,
            "filename": filename,
            "file_code": normalize_file_code(filename),
            "description": description,
            "url": full_url,
        })

    unique = {}
    for r in records:
        unique[r["url"]] = r
    return list(unique.values())


def download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return False

    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name[:55], leave=False) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return True


def normalize_file_code(path_or_name) -> str:
    stem = Path(str(path_or_name)).stem.upper()
    stem = re.sub(r"^P_", "", stem)
    stem = re.sub(r"_[A-Z]$", "", stem)
    return stem


def infer_cycle_from_path(path: Path) -> str:
    text = str(path).lower()
    stem = path.stem.upper()

    if re.search(r"2017.*2020|pre.*pandemic|prepandemic", text) or stem.startswith("P_"):
        return "2017-March_2020_PrePandemic"
    if re.search(r"2021.*2023|august.*2023|2021-2023", text) or stem.endswith("_L"):
        return "2021-August_2023"
    if re.search(r"2025.*2026|2025-2026", text) or stem.endswith("_M"):
        return "2025-2026"
    return "Unknown_Cycle"


def download_missing_core_and_targets():
    """
    Download missing DEMO/PFQ/WHQ/HSQ files if available from CDC.
    PFQ may not be available in later releases; WHQ/HSQ are fallback targets.
    """
    print("\n================ CHECKING CDC FOR MISSING TARGET FILES ================\n")

    wanted_codes = {"DEMO", "PFQ", "WHQ", "HSQ"}

    log = []
    for cycle_label, cycle_code in TARGET_CYCLES.items():
        for component in ["Demographics", "Questionnaire"]:
            try:
                links = discover_xpt_links(cycle_label, cycle_code, component)
            except Exception as exc:
                print(f"Could not inspect {cycle_label} {component}: {exc}")
                continue

            for rec in links:
                if rec["file_code"] not in wanted_codes:
                    continue

                dest = RAW_DIR / cycle_label / component / rec["filename"]
                try:
                    downloaded = download_file(rec["url"], dest)
                    status = "downloaded" if downloaded else "already_exists"
                    print(f"{status}: {cycle_label} / {component} / {rec['filename']}")
                    log.append({**rec, "local_path": str(dest), "status": status})
                except Exception as exc:
                    print(f"FAILED: {rec['url']} -> {exc}")
                    log.append({**rec, "local_path": str(dest), "status": "failed", "error": str(exc)})

    log_df = pd.DataFrame(log)
    log_path = OUT_DIR / "missing_core_target_download_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved missing-core download log: {log_path}")


# ============================================================
# 3) DATA HELPERS
# ============================================================

def read_xpt(path: Path) -> pd.DataFrame:
    try:
        return pd.read_sas(path, format="xport", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_sas(path, format="xport", encoding="latin1")


def keep_existing(df: pd.DataFrame, cols: List[Optional[str]]) -> pd.DataFrame:
    cols = [c for c in cols if c and c in df.columns]
    return df[cols].copy()


def replace_special_missing(series: pd.Series) -> pd.Series:
    return series.replace({
        5: np.nan, 7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan,
        777: np.nan, 999: np.nan, 7777: np.nan, 9999: np.nan,
        77777: np.nan, 99999: np.nan,
    })


def numeric_clean(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = replace_special_missing(df[c])
    return df


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fix_tiny_values_to_zero(series: pd.Series, threshold: float = 1e-10) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    return series.mask(series.abs() < threshold, 0)


def mean_with_min_valid(row: pd.Series, cols: List[str], min_valid: int = 2) -> float:
    if not cols:
        return np.nan
    vals = pd.to_numeric(row[cols], errors="coerce").replace(0, np.nan)
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


def build_inventory() -> pd.DataFrame:
    xpt_files = sorted(RAW_DIR.rglob("*.xpt")) + sorted(RAW_DIR.rglob("*.XPT"))
    rows = []
    for p in xpt_files:
        rows.append({
            "cycle_inferred": infer_cycle_from_path(p),
            "component_folder": p.parent.name,
            "normalized_code": normalize_file_code(p),
            "filename": p.name,
            "relative_path": str(p.relative_to(RAW_DIR)),
            "full_path": str(p),
            "file_size_bytes": p.stat().st_size,
        })

    inv = pd.DataFrame(rows).drop_duplicates(subset=["full_path"]).reset_index(drop=True) if rows else pd.DataFrame()
    inv_path = OUT_DIR / "later_cycles_xpt_inventory_after_core_download.csv"
    inv.to_csv(inv_path, index=False, encoding="utf-8-sig")
    print(f"Saved inventory: {inv_path}")
    return inv


def find_cycle_files(inventory: pd.DataFrame, cycle_label: str) -> Dict[str, Path]:
    cycle_df = inventory[inventory["cycle_inferred"] == cycle_label].copy()
    files_map = {}

    for key, possible_codes in PROJECT_CODES.items():
        matches = cycle_df[cycle_df["normalized_code"].isin([c.upper() for c in possible_codes])]
        if matches.empty:
            continue

        paths = [Path(p) for p in matches["full_path"].tolist()]
        paths = sorted(paths, key=lambda p: (len(p.name), p.name))
        files_map[key] = paths[0]

    return files_map


# ============================================================
# 4) PREPARE COMPONENTS
# ============================================================

def empty_by_seqn(demo: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = demo[["SEQN"]].copy()
    for c in cols:
        out[c] = np.nan
    return out


def prepare_demo(path: Path) -> pd.DataFrame:
    df = read_xpt(path)
    df = keep_existing(df, ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "RIDRETH3", "DMDEDUC2", "INDFMPIR"])
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    df["age"] = df.get("RIDAGEYR", np.nan)
    df["sex"] = df.get("RIAGENDR", np.nan)
    df["race_ethnicity"] = df["RIDRETH1"] if "RIDRETH1" in df.columns else df.get("RIDRETH3", np.nan)
    df["education"] = df.get("DMDEDUC2", np.nan)
    df["income_poverty_ratio"] = fix_tiny_values_to_zero(df.get("INDFMPIR", np.nan))
    return df[["SEQN", "age", "sex", "race_ethnicity", "education", "income_poverty_ratio"]]


def prepare_bmx(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    if not path:
        return empty_by_seqn(demo, ["BMI", "height_cm", "weight_kg"])
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


def prepare_bpx(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    if not path:
        return empty_by_seqn(demo, ["mean_SBP", "mean_DBP"])
    df = read_xpt(path)
    df = keep_existing(df, ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])
    sbp = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in df.columns]
    dbp = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in df.columns]
    df["mean_SBP"] = df.apply(lambda r: mean_with_min_valid(r, sbp), axis=1)
    df["mean_DBP"] = df.apply(lambda r: mean_with_min_valid(r, dbp), axis=1)
    df.loc[(df["mean_SBP"] < 70) | (df["mean_SBP"] > 250), "mean_SBP"] = np.nan
    df.loc[(df["mean_DBP"] < 40) | (df["mean_DBP"] > 150), "mean_DBP"] = np.nan
    return df[["SEQN", "mean_SBP", "mean_DBP"]]


def prepare_lab(path: Optional[Path], demo: pd.DataFrame, candidates: List[str], out_col: str, low=None, high=None) -> pd.DataFrame:
    if not path:
        return empty_by_seqn(demo, [out_col])
    df = read_xpt(path)
    df = keep_existing(df, ["SEQN"] + candidates)
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])
    col = first_existing(df, candidates)
    df[out_col] = df[col] if col else np.nan
    if low is not None:
        df.loc[df[out_col] < low, out_col] = np.nan
    if high is not None:
        df.loc[df[out_col] > high, out_col] = np.nan
    return df[["SEQN", out_col]]


def prepare_mgx(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    if not path:
        return empty_by_seqn(demo, ["max_grip_strength"])
    df = read_xpt(path)
    grip = ["MGXH1T1", "MGXH1T2", "MGXH1T3", "MGXH2T1", "MGXH2T2", "MGXH2T3"]
    df = keep_existing(df, ["SEQN"] + grip)
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])
    existing = [c for c in grip if c in df.columns]
    if existing:
        for c in existing:
            df.loc[(df[c] < 1) | (df[c] > 100), c] = np.nan
        df["max_grip_strength"] = df[existing].max(axis=1, skipna=True)
    else:
        df["max_grip_strength"] = np.nan
    return df[["SEQN", "max_grip_strength"]]


def prepare_pfq_or_fallback(files: Dict[str, Path], demo: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Primary target: disability_stage from PFQ061 variables.
    Fallback 1: functional limitation from WHQ if suitable variables are present.
    Fallback 2: health-status risk from HSQ if no functional limitation is available.
    """
    out_cols = ["upper_limb_difficulty", "mobility_difficulty", "adl_difficulty", "disability_stage",
                "fallback_function_risk", "self_rated_health_risk"]

    # ---------- PFQ primary ----------
    if "PFQ" in files:
        df = read_xpt(files["PFQ"])
        old = ["PFQ061P", "PFQ061E", "PFQ061B", "PFQ061C", "PFQ061M", "PFQ061K", "PFQ061L"]
        df = keep_existing(df, ["SEQN"] + old + [c for c in df.columns if c.startswith("PFQ")])
        df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
        numeric_clean(df, [c for c in df.columns if c != "SEQN"])

        existing = [c for c in old if c in df.columns]
        if len(existing) >= 3:
            def difficulty_binary(x):
                if pd.isna(x): return np.nan
                if x == 1: return 0
                if x in [2, 3, 4]: return 1
                return np.nan

            for c in existing:
                df[c + "_bin"] = df[c].apply(difficulty_binary)

            def max_existing(cols):
                cols = [c for c in cols if c in df.columns]
                return df[cols].max(axis=1, skipna=True) if cols else np.nan

            df["upper_limb_difficulty"] = max_existing(["PFQ061P_bin", "PFQ061E_bin"])
            df["mobility_difficulty"] = max_existing(["PFQ061B_bin", "PFQ061C_bin", "PFQ061M_bin"])
            df["adl_difficulty"] = max_existing(["PFQ061K_bin", "PFQ061L_bin"])

            def assign_stage(row):
                vals = [row[c + "_bin"] for c in existing if c + "_bin" in row.index]
                if all(pd.isna(v) for v in vals): return np.nan
                if row.get("adl_difficulty", np.nan) == 1: return 3
                if row.get("mobility_difficulty", np.nan) == 1: return 2
                if row.get("upper_limb_difficulty", np.nan) == 1: return 1
                observed = [v for v in vals if not pd.isna(v)]
                if observed and max(observed) == 0: return 0
                return np.nan

            df["disability_stage"] = df.apply(assign_stage, axis=1)
            df["fallback_function_risk"] = np.nan
            df["self_rated_health_risk"] = np.nan
            return df[["SEQN"] + out_cols], "PFQ_disability_stage"

    # ---------- WHQ fallback ----------
    if "WHQ" in files:
        df = read_xpt(files["WHQ"])
        whq_cols = [c for c in df.columns if c.startswith("WHQ")]
        df = keep_existing(df, ["SEQN"] + whq_cols)
        df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
        numeric_clean(df, [c for c in df.columns if c != "SEQN"])

        # General fallback: use any WHQ binary/ordinal difficulty-like columns if present.
        # This is intentionally conservative; inspect the report before manuscript use.
        candidate_cols = [c for c in whq_cols if c in df.columns and df[c].dropna().isin([1, 2, 3, 4]).mean() > 0.5]
        if candidate_cols:
            # Treat 1 as no/low and 2-4 as higher risk only as a fallback marker.
            bins = []
            for c in candidate_cols:
                b = c + "_riskbin"
                df[b] = np.where(df[c].isna(), np.nan, np.where(df[c].isin([2, 3, 4]), 1, 0))
                bins.append(b)
            df["fallback_function_risk"] = df[bins].max(axis=1, skipna=True)
        else:
            df["fallback_function_risk"] = np.nan

        df["upper_limb_difficulty"] = np.nan
        df["mobility_difficulty"] = np.nan
        df["adl_difficulty"] = np.nan
        df["disability_stage"] = df["fallback_function_risk"]
        df["self_rated_health_risk"] = np.nan
        return df[["SEQN"] + out_cols], "WHQ_fallback_function_risk"

    # ---------- HSQ fallback ----------
    if "HSQ" in files:
        df = read_xpt(files["HSQ"])
        hsq_cols = [c for c in df.columns if c.startswith("HSQ") or c.startswith("HSD")]
        df = keep_existing(df, ["SEQN"] + hsq_cols)
        df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
        numeric_clean(df, [c for c in df.columns if c != "SEQN"])

        # Common NHANES self-rated general health variable: HSD010
        if "HSD010" in df.columns:
            # 1 excellent, 2 very good, 3 good, 4 fair, 5 poor.
            df["self_rated_health_risk"] = np.where(df["HSD010"].isna(), np.nan, np.where(df["HSD010"].isin([4, 5]), 1, 0))
        else:
            df["self_rated_health_risk"] = np.nan

        df["upper_limb_difficulty"] = np.nan
        df["mobility_difficulty"] = np.nan
        df["adl_difficulty"] = np.nan
        df["fallback_function_risk"] = np.nan
        df["disability_stage"] = df["self_rated_health_risk"]
        return df[["SEQN"] + out_cols], "HSQ_fallback_self_rated_health_risk"

    out = empty_by_seqn(demo, out_cols)
    return out, "no_disability_or_fallback_target"


def prepare_smq(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    if not path:
        return empty_by_seqn(demo, ["smoking_status"])
    df = read_xpt(path)
    df = keep_existing(df, ["SEQN", "SMQ020", "SMQ040"])
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    def smoking_status(row):
        ever = row.get("SMQ020", np.nan)
        now = row.get("SMQ040", np.nan)
        if pd.isna(ever): return np.nan
        if ever == 2: return 0
        if ever == 1:
            if now in [1, 2]: return 2
            if now == 3: return 1
            return 1
        return np.nan

    df["smoking_status"] = df.apply(smoking_status, axis=1)
    return df[["SEQN", "smoking_status"]]


def prepare_paq(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    if not path:
        return empty_by_seqn(demo, ["MET_min_week", "meets_activity_guideline", "sedentary_minutes_day"])
    df = read_xpt(path)
    cols = ["SEQN", "PAQ605", "PAQ610", "PAD615", "PAQ620", "PAQ625", "PAD630",
            "PAQ635", "PAQ640", "PAD645", "PAQ650", "PAQ655", "PAD660",
            "PAQ665", "PAQ670", "PAD675", "PAD680"]
    df = keep_existing(df, cols)
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    def comp(gate, days, mins, met):
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        if gate not in df.columns: return out
        yes = df[gate] == 1
        no = df[gate] == 2
        if days in df.columns and mins in df.columns:
            idx = yes & df[days].notna() & df[mins].notna()
            out.loc[idx] = df.loc[idx, days] * df.loc[idx, mins] * met
        out.loc[no] = 0
        return out

    df["vig_work_met"] = comp("PAQ605", "PAQ610", "PAD615", 8)
    df["mod_work_met"] = comp("PAQ620", "PAQ625", "PAD630", 4)
    df["transport_met"] = comp("PAQ635", "PAQ640", "PAD645", 4)
    df["vig_rec_met"] = comp("PAQ650", "PAQ655", "PAD660", 8)
    df["mod_rec_met"] = comp("PAQ665", "PAQ670", "PAD675", 4)
    met_cols = ["vig_work_met", "mod_work_met", "transport_met", "vig_rec_met", "mod_rec_met"]
    df["MET_min_week"] = fix_tiny_values_to_zero(df[met_cols].sum(axis=1, min_count=1))
    df["meets_activity_guideline"] = np.where(df["MET_min_week"].isna(), np.nan, np.where(df["MET_min_week"] >= 600, 1, 0))
    df["sedentary_minutes_day"] = df["PAD680"] if "PAD680" in df.columns else np.nan
    df.loc[(df["sedentary_minutes_day"] < 0) | (df["sedentary_minutes_day"] > 1440), "sedentary_minutes_day"] = np.nan
    return df[["SEQN", "MET_min_week", "meets_activity_guideline", "sedentary_minutes_day"]]


def prepare_dpq(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    if not path:
        return empty_by_seqn(demo, ["PHQ9_score", "PHQ9_category"])
    df = read_xpt(path)
    items = ["DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090"]
    df = keep_existing(df, ["SEQN"] + items)
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])

    def phq_category(score):
        if pd.isna(score): return np.nan
        if score < 5: return 0
        if score < 10: return 1
        if score < 15: return 2
        if score < 20: return 3
        return 4

    df["PHQ9_score"] = df.apply(lambda r: sum_with_all_required(r, items), axis=1) if all(c in df.columns for c in items) else np.nan
    df["PHQ9_score"] = fix_tiny_values_to_zero(df["PHQ9_score"])
    df["PHQ9_category"] = df["PHQ9_score"].apply(phq_category)
    return df[["SEQN", "PHQ9_score", "PHQ9_category"]]


def prepare_mcq(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    if not path:
        return empty_by_seqn(demo, ["arthritis", "arthritis_type"])
    df = read_xpt(path)
    df = keep_existing(df, ["SEQN", "MCQ160A", "MCQ195"])
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    numeric_clean(df, [c for c in df.columns if c != "SEQN"])
    df["arthritis"] = np.where(df.get("MCQ160A", np.nan).isna(), np.nan,
                               np.where(df["MCQ160A"] == 1, 1, np.where(df["MCQ160A"] == 2, 0, np.nan)))
    df["arthritis_type"] = df["MCQ195"] if "MCQ195" in df.columns else np.nan
    return df[["SEQN", "arthritis", "arthritis_type"]]


def prepare_rxq(path: Optional[Path], demo: pd.DataFrame) -> pd.DataFrame:
    out = empty_by_seqn(demo, ["medication_count", "polypharmacy"])
    if not path:
        return out
    df = read_xpt(path)
    drug_col = first_existing(df, ["RXDDRUG", "RXDDRGID", "RXDRSC1", "RXDRSC2", "RXDRSC3"])
    df = keep_existing(df, ["SEQN", drug_col])
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce")
    if drug_col is None:
        return out
    df = df[df[drug_col].notna()].copy()
    if df.empty:
        return out
    agg = df.groupby("SEQN", as_index=False)[drug_col].nunique()
    agg.rename(columns={drug_col: "medication_count"}, inplace=True)
    agg["polypharmacy"] = np.where(agg["medication_count"] >= 5, 1, 0)
    return out.drop(columns=["medication_count", "polypharmacy"]).merge(agg, on="SEQN", how="left")


# ============================================================
# 5) PROCESS CYCLES
# ============================================================

def process_cycle(cycle_label: str, files: Dict[str, Path]) -> Optional[pd.DataFrame]:
    print(f"\n================ PROCESSING {cycle_label} ================\n")

    if "DEMO" not in files:
        print(f"SKIPPED {cycle_label}: DEMO still missing.")
        return None

    print("Files available for processing:")
    for k, v in sorted(files.items()):
        print(f"  {k}: {v.name}")

    demo = prepare_demo(files["DEMO"])
    bmx = prepare_bmx(files.get("BMX"), demo)
    bpx = prepare_bpx(files.get("BPX"), demo)
    mgx = prepare_mgx(files.get("MGX"), demo)

    ghb = prepare_lab(files.get("GHB"), demo, ["LBXGH"], "HbA1c", 3, 20)
    glu = prepare_lab(files.get("GLU"), demo, ["LBXGLU"], "fasting_glucose", 20, 800)
    tchol = prepare_lab(files.get("TCHOL"), demo, ["LBXTC"], "total_cholesterol", 50, 500)
    hdl = prepare_lab(files.get("HDL"), demo, ["LBDHDD", "LBXHDD"], "HDL", 10, 200)
    trig = prepare_lab(files.get("TRIGLY"), demo, ["LBXTR"], "triglycerides", 20, 3000)

    target_df, target_source = prepare_pfq_or_fallback(files, demo)

    smq = prepare_smq(files.get("SMQ"), demo)
    paq = prepare_paq(files.get("PAQ"), demo)
    dpq = prepare_dpq(files.get("DPQ"), demo)
    mcq = prepare_mcq(files.get("MCQ"), demo)
    rxq = prepare_rxq(files.get("RXQ_RX"), demo)

    dfs = [demo, bmx, bpx, mgx, ghb, glu, tchol, hdl, trig, target_df, smq, paq, dpq, mcq, rxq]

    final_df = dfs[0]
    for d in dfs[1:]:
        final_df = final_df.merge(d, on="SEQN", how="left")

    final_df["LDL"] = np.where(
        final_df["triglycerides"].notna() & final_df["total_cholesterol"].notna()
        & final_df["HDL"].notna() & (final_df["triglycerides"] < 400),
        final_df["total_cholesterol"] - final_df["HDL"] - (final_df["triglycerides"] / 5.0),
        np.nan
    )

    for col in ["PHQ9_score", "income_poverty_ratio", "MET_min_week", "mean_SBP", "mean_DBP", "BMI"]:
        if col in final_df.columns:
            final_df[col] = fix_tiny_values_to_zero(final_df[col])

    final_df = final_df[final_df["age"] >= 20].copy()
    final_df["cycle"] = cycle_label
    final_df["target_source"] = target_source

    preferred = [
        "SEQN", "cycle", "target_source",
        "disability_stage", "upper_limb_difficulty", "mobility_difficulty", "adl_difficulty",
        "fallback_function_risk", "self_rated_health_risk",
        "age", "sex", "race_ethnicity", "education", "income_poverty_ratio",
        "BMI", "height_cm", "weight_kg", "mean_SBP", "mean_DBP", "max_grip_strength",
        "HbA1c", "fasting_glucose", "total_cholesterol", "HDL", "triglycerides", "LDL",
        "smoking_status", "MET_min_week", "meets_activity_guideline", "sedentary_minutes_day",
        "PHQ9_score", "PHQ9_category", "arthritis", "arthritis_type",
        "medication_count", "polypharmacy",
    ]
    final_df = final_df[[c for c in preferred if c in final_df.columns]]

    out_all = OUT_DIR / f"NHANES_{cycle_label}_processed_all_adults.csv"
    final_df.to_csv(out_all, index=False, encoding="utf-8-sig")

    target_col = "disability_stage"
    filtered = final_df[final_df[target_col].notna()].copy() if target_col in final_df.columns else final_df.iloc[0:0].copy()
    out_final = OUT_DIR / f"NHANES_{cycle_label}_final_processed.csv"
    filtered.to_csv(out_final, index=False, encoding="utf-8-sig")

    report = {
        "cycle_label": cycle_label,
        "target_source": target_source,
        "all_adults_shape": list(final_df.shape),
        "final_filtered_shape": list(filtered.shape),
        "files_used": {k: str(v) for k, v in files.items()},
        "target_counts": filtered[target_col].value_counts(dropna=False).sort_index().to_dict() if not filtered.empty else {},
        "missingness_percent_all_adults": (final_df.isna().mean() * 100).round(1).sort_values(ascending=False).to_dict(),
        "saved_all": str(out_all),
        "saved_final": str(out_final),
    }
    (OUT_DIR / f"{cycle_label}_processing_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Target source: {target_source}")
    print(f"Saved all adults: {out_all}")
    print(f"Saved final filtered: {out_final}")
    print(f"All adults shape: {final_df.shape}")
    print(f"Final filtered shape: {filtered.shape}")

    return filtered


def main():
    print("NHANES missing-core downloader and later-cycle processor")
    print(f"Raw input folder: {RAW_DIR}")
    print(f"Output folder: {OUT_DIR}")

    # Step A: download missing DEMO and fallback target files if available.
    download_missing_core_and_targets()

    # Step B: rebuild inventory after downloads.
    inv = build_inventory()

    if inv.empty:
        print("No XPT files found. Stop.")
        return

    print("\n================ UPDATED FOLDER SUMMARY ================\n")
    summary = inv.groupby(["cycle_inferred", "component_folder"], as_index=False).agg(files=("filename", "count"))
    print(summary.to_string(index=False))

    # Step C: process known cycles.
    processed = []
    log_rows = []

    for cycle_label in sorted([c for c in inv["cycle_inferred"].unique() if c != "Unknown_Cycle"]):
        files = find_cycle_files(inv, cycle_label)
        df = process_cycle(cycle_label, files)

        log_rows.append({
            "cycle_label": cycle_label,
            "processed": df is not None,
            "rows": int(df.shape[0]) if df is not None else 0,
            "columns": int(df.shape[1]) if df is not None else 0,
            "available_keys": ", ".join(sorted(files.keys())),
            "has_DEMO": "DEMO" in files,
            "has_PFQ": "PFQ" in files,
            "has_WHQ": "WHQ" in files,
            "has_HSQ": "HSQ" in files,
        })

        if df is not None and not df.empty:
            processed.append(df)

    log_df = pd.DataFrame(log_rows)
    log_path = OUT_DIR / "later_cycles_processing_log_after_core_download.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved processing log: {log_path}")

    if processed:
        pooled = pd.concat(processed, ignore_index=True)
        pooled_path = OUT_DIR / "NHANES_later_cycles_pooled_final_processed.csv"
        pooled.to_csv(pooled_path, index=False, encoding="utf-8-sig")
        print(f"Saved pooled later-cycle final file: {pooled_path}")
        print(f"Pooled shape: {pooled.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
NHANES 2019-2026 Raw Data Explorer and Downloader
=================================================

Purpose
-------
Explore CDC NHANES data pages and download available XPT files for later
NHANES releases after the 2011-2018 development data.

Important notes
---------------
1. NHANES data are released by survey cycle/release period, not by exact year.
2. The public post-2018 releases currently include:
   - 2017-March 2020 Pre-Pandemic: CDC cycle code "2017-2020"; file prefix often "P_"
   - August 2021-August 2023: CDC cycle code "2021-2023"; file suffix often "_L"
   - 2025-2026: portal exists, but many files may be documentation/release pages only until data files are published.
3. The 2019-March 2020 convenience sample has limitations and should not be treated as a standard nationally representative 2019-2020 yearly cycle.
4. This script downloads RAW XPT files and creates an availability manifest.
   Processing/harmonization into final analytical CSV files should be done in a later step.

Destination folder
------------------
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/data_raw
Requirements
------------
pip install requests beautifulsoup4 pandas tqdm
"""

import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ============================================================
# 0) USER SETTINGS
# ============================================================

BASE_DIR = Path(r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\data_raw")

# NHANES release periods to explore.
# These are not exact calendar years; they are CDC release/cycle identifiers.
TARGET_CYCLES = {
    "2017-March_2020_PrePandemic": "2017-2020",
    "2021-August_2023": "2021-2023",
    "2025-2026": "2025-2026",
}

# Components to explore.
COMPONENTS = [
    "Demographics",
    "Dietary",
    "Examination",
    "Laboratory",
    "Questionnaire",
]

# Download mode:
#   "all"      -> download every discovered XPT file.
#   "selected" -> download only likely files needed for this disability-health paper.
DOWNLOAD_MODE = "selected"

# Keywords/file-code patterns likely relevant to this project.
# The script searches both the file code and visible description text.
SELECTED_PATTERNS = [
    # Core demographics
    r"\bDEMO\b",

    # Body measures / blood pressure / grip strength
    r"\bBMX\b",
    r"\bBPX\b",
    r"\bMGX\b",
    r"Body Measures",
    r"Blood Pressure",
    r"Muscle Strength",
    r"Grip",

    # Laboratory/metabolic markers
    r"\bGHB\b",
    r"\bGLU\b",
    r"\bTCHOL\b",
    r"\bHDL\b",
    r"\bTRIGLY\b",
    r"\bLDL\b",
    r"Glycohemoglobin",
    r"Glucose",
    r"Cholesterol",
    r"Triglycerides",

    # Functioning/disability and physical function
    r"\bPFQ\b",
    r"\bFNQ\b",
    r"Functioning",
    r"Physical Functioning",
    r"Disability",

    # Behavior, mental health, medical conditions, medication
    r"\bSMQ\b",
    r"\bPAQ\b",
    r"\bDPQ\b",
    r"\bMCQ\b",
    r"\bRXQ_RX\b",
    r"\bRXQ\b",
    r"Smoking",
    r"Physical Activity",
    r"Depression",
    r"Medical Conditions",
    r"Prescription Medications",

    # Nutrition-related questionnaires/dietary files, optional for future extension
    r"\bDR1TOT\b",
    r"\bDR2TOT\b",
    r"\bDSQ\b",
    r"Dietary Interview",
    r"Total Nutrient",
    r"Supplement",
]


# ============================================================
# 1) GENERAL HELPERS
# ============================================================

CDC_BASE = "https://wwwn.cdc.gov/nchs/nhanes/"
SEARCH_PAGE = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx"
CONTINUOUS_PAGE = "https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

session = requests.Session()
session.headers.update(HEADERS)


def safe_filename(name: str) -> str:
    """Create a filesystem-safe filename."""
    name = unquote(name)
    name = re.sub(r"[<>:\"/\\|?*]+", "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name


def get_html(url: str, timeout: int = 30) -> str:
    """Download HTML with a basic retry mechanism."""
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


def discover_xpt_links_from_page(url: str, cycle_label: str, cycle_code: str, component: str):
    """
    Discover XPT links from a CDC NHANES data page.

    Returns list of dicts:
      cycle_label, cycle_code, component, file_code, description, url, filename
    """
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = " ".join(a.get_text(" ", strip=True).split())

        # Keep direct XPT links only.
        if ".XPT" not in href.upper() and "Data [XPT" not in text:
            continue

        full_url = urljoin(url, href)

        # Extract filename from URL path.
        parsed = urlparse(full_url)
        filename = safe_filename(os.path.basename(parsed.path))
        if not filename.lower().endswith(".xpt"):
            # Try to extract from href directly.
            filename = safe_filename(os.path.basename(href))
        if not filename.lower().endswith(".xpt"):
            continue

        # File code from filename.
        file_code = Path(filename).stem

        # Try to infer row/nearby text for description.
        tr = a.find_parent("tr")
        if tr is not None:
            description = " ".join(tr.get_text(" ", strip=True).split())
        else:
            parent = a.find_parent()
            description = " ".join(parent.get_text(" ", strip=True).split()) if parent else text

        rows.append({
            "cycle_label": cycle_label,
            "cycle_code": cycle_code,
            "component": component,
            "file_code": file_code,
            "description": description,
            "url": full_url,
            "filename": filename,
        })

    # De-duplicate by URL.
    unique = {}
    for r in rows:
        unique[r["url"]] = r
    return list(unique.values())


def nhanes_component_url(cycle_code: str, component: str) -> str:
    return f"{SEARCH_PAGE}?Component={component}&Cycle={cycle_code}"


def nhanes_cycle_main_url(cycle_code: str) -> str:
    return f"{CONTINUOUS_PAGE}?Cycle={cycle_code}"


def is_selected(record: dict) -> bool:
    haystack = f"{record.get('file_code','')} {record.get('description','')}"
    return any(re.search(pattern, haystack, flags=re.IGNORECASE) for pattern in SELECTED_PATTERNS)


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> bool:
    """Download URL to dest. Returns True if downloaded, False if skipped because file exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return False

    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name[:60],
            leave=False,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return True


# ============================================================
# 2) EXPLORE NHANES CYCLES
# ============================================================

def explore_cycles():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    cycle_status_rows = []

    for cycle_label, cycle_code in TARGET_CYCLES.items():
        print(f"\n=== Exploring NHANES cycle/release: {cycle_label} ({cycle_code}) ===")

        # Save main page HTML for traceability.
        main_url = nhanes_cycle_main_url(cycle_code)
        try:
            main_html = get_html(main_url)
            cycle_dir = BASE_DIR / cycle_label
            cycle_dir.mkdir(parents=True, exist_ok=True)
            (cycle_dir / "cycle_main_page.html").write_text(main_html, encoding="utf-8")
            main_ok = True
        except Exception as exc:
            print(f"Could not fetch main cycle page: {exc}")
            main_ok = False

        for component in COMPONENTS:
            page_url = nhanes_component_url(cycle_code, component)
            try:
                records = discover_xpt_links_from_page(
                    url=page_url,
                    cycle_label=cycle_label,
                    cycle_code=cycle_code,
                    component=component,
                )
                print(f"{component}: {len(records)} XPT links found")
                all_records.extend(records)

                cycle_status_rows.append({
                    "cycle_label": cycle_label,
                    "cycle_code": cycle_code,
                    "component": component,
                    "page_url": page_url,
                    "xpt_links_found": len(records),
                    "page_accessible": True,
                    "main_page_accessible": main_ok,
                })

            except Exception as exc:
                print(f"{component}: FAILED -> {exc}")
                cycle_status_rows.append({
                    "cycle_label": cycle_label,
                    "cycle_code": cycle_code,
                    "component": component,
                    "page_url": page_url,
                    "xpt_links_found": 0,
                    "page_accessible": False,
                    "main_page_accessible": main_ok,
                    "error": str(exc),
                })

    manifest = pd.DataFrame(all_records)
    status = pd.DataFrame(cycle_status_rows)

    if not manifest.empty:
        manifest["selected_for_download"] = manifest.apply(is_selected, axis=1)
        manifest = manifest.sort_values(["cycle_label", "component", "file_code"]).reset_index(drop=True)

    manifest_path = BASE_DIR / "NHANES_2019_2026_xpt_availability_manifest.csv"
    status_path = BASE_DIR / "NHANES_2019_2026_cycle_component_status.csv"

    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    status.to_csv(status_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved manifest: {manifest_path}")
    print(f"Saved status report: {status_path}")

    return manifest, status


# ============================================================
# 3) DOWNLOAD AVAILABLE XPT FILES
# ============================================================

def download_selected_files(manifest: pd.DataFrame):
    if manifest.empty:
        print("No XPT files discovered. Nothing to download.")
        return pd.DataFrame()

    if DOWNLOAD_MODE.lower() == "all":
        to_download = manifest.copy()
    else:
        to_download = manifest[manifest["selected_for_download"]].copy()

    print(f"\nDownload mode: {DOWNLOAD_MODE}")
    print(f"Files selected for download: {len(to_download)}")

    log_rows = []

    for _, row in to_download.iterrows():
        cycle_dir = BASE_DIR / row["cycle_label"] / row["component"]
        dest = cycle_dir / row["filename"]

        try:
            downloaded = download_file(row["url"], dest)
            status = "downloaded" if downloaded else "already_exists"
            print(f"{status}: {row['cycle_label']} / {row['component']} / {row['filename']}")

            log_rows.append({
                **row.to_dict(),
                "local_path": str(dest),
                "download_status": status,
                "file_size_bytes": dest.stat().st_size if dest.exists() else None,
            })

        except Exception as exc:
            print(f"FAILED: {row['url']} -> {exc}")
            log_rows.append({
                **row.to_dict(),
                "local_path": str(dest),
                "download_status": "failed",
                "error": str(exc),
                "file_size_bytes": None,
            })

    log = pd.DataFrame(log_rows)
    log_path = BASE_DIR / "NHANES_2019_2026_download_log.csv"
    log.to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved download log: {log_path}")

    return log


# ============================================================
# 4) OPTIONAL: QUICK INVENTORY SUMMARY
# ============================================================

def summarize_downloads(download_log: pd.DataFrame, manifest: pd.DataFrame):
    summary_rows = []

    if not manifest.empty:
        summary = (
            manifest.groupby(["cycle_label", "component"], as_index=False)
            .agg(
                discovered_xpt_files=("url", "nunique"),
                selected_xpt_files=("selected_for_download", "sum"),
            )
        )
        summary_rows.append(summary)

    if not download_log.empty:
        dsum = (
            download_log.groupby(["cycle_label", "component", "download_status"], as_index=False)
            .agg(downloaded_files=("url", "nunique"))
        )
        dsum_path = BASE_DIR / "NHANES_2019_2026_download_summary_by_status.csv"
        dsum.to_csv(dsum_path, index=False, encoding="utf-8-sig")
        print(f"Saved download summary by status: {dsum_path}")

    if summary_rows:
        final_summary = summary_rows[0]
        summary_path = BASE_DIR / "NHANES_2019_2026_inventory_summary.csv"
        final_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Saved inventory summary: {summary_path}")

        print("\nInventory summary:")
        print(final_summary.to_string(index=False))


# ============================================================
# 5) MAIN
# ============================================================

if __name__ == "__main__":
    print("NHANES 2019-2026 raw data explorer/downloader")
    print(f"Destination: {BASE_DIR}")

    manifest_df, status_df = explore_cycles()
    download_log_df = download_selected_files(manifest_df)
    summarize_downloads(download_log_df, manifest_df)

    print("\nDone.")
    print("Next step: inspect the manifest and download log, then harmonize variables into processed cycle-level CSV files.")

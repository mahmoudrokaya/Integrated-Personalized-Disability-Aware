# -*- coding: utf-8 -*-
"""
scan_experiments_folder.py

Purpose
-------
Scan the full experiments folder structure and generate a clean inventory of:
1. folders
2. files
3. file sizes
4. file extensions
5. likely experiment outputs
6. result tables
7. figures
8. markdown summaries
9. model checkpoints
10. logs / reports

This inventory will help decide which files should be uploaded before writing
each part of Section 4.

Root folder:
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments
"""

import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


ROOT = Path(r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments")
OUTPUT_DIR = ROOT / "_folder_inventory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TREE_DEPTH = 5


def human_size(num_bytes):
    if num_bytes is None:
        return ""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def classify_file(path: Path):
    name = path.name.lower()
    ext = path.suffix.lower()

    if ext in [".csv", ".xlsx", ".xls"]:
        return "table_or_dataset"
    if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".svg", ".pdf"]:
        return "figure_or_visual"
    if ext in [".md", ".txt", ".json", ".log"]:
        return "report_or_metadata"
    if ext in [".pt", ".pth", ".pkl", ".joblib"]:
        return "model_or_checkpoint"
    if ext in [".py", ".ipynb"]:
        return "code"
    return "other"


def guess_experiment(path: Path):
    text = str(path).lower()

    for i in range(1, 6):
        if f"experiment{i}" in text or f"experiment{i}" in path.name.lower():
            return f"Experiment{i}"

    if "modeling_ready" in text:
        return "ModelingReady"
    if "pre-processed-cycles" in text or "pre_processed" in text or "preprocessed" in text:
        return "Preprocessing"
    if "data_processed_later_cycles" in text:
        return "LaterCyclePreprocessing"
    if "data_raw" in text:
        return "RawData"
    if "codes" in text:
        return "Codes"

    return "General"


def is_likely_key_output(path: Path):
    name = path.name.lower()
    ext = path.suffix.lower()

    keywords = [
        "metrics",
        "summary",
        "report",
        "confusion",
        "classification",
        "weights",
        "convergence",
        "latency",
        "efficiency",
        "availability",
        "shift",
        "target_distribution",
        "metadata",
        "paper_ready",
        "model_size",
        "madds",
        "fairness",
        "explainability",
        "permutation",
        "ablation",
        "distribution",
        "missing",
        "audit",
        "logical",
    ]

    if any(k in name for k in keywords):
        return True

    if ext in [".png", ".jpg", ".jpeg"] and name.startswith("fig"):
        return True

    return False


def scan_files(root: Path):
    rows = []
    folder_rows = []

    for current, dirs, files in os.walk(root):
        current_path = Path(current)

        rel_folder = current_path.relative_to(root)
        depth = len(rel_folder.parts)

        folder_rows.append({
            "folder": str(current_path),
            "relative_folder": str(rel_folder),
            "depth": depth,
            "num_subfolders": len(dirs),
            "num_files": len(files),
            "experiment_group": guess_experiment(current_path),
        })

        for file in files:
            path = current_path / file
            try:
                stat = path.stat()
                size = stat.st_size
                modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
            except Exception:
                size = None
                modified = ""

            rows.append({
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "file_path": str(path),
                "relative_path": str(path.relative_to(root)),
                "parent_folder": str(current_path),
                "relative_parent_folder": str(rel_folder),
                "size_bytes": size,
                "size_readable": human_size(size),
                "modified_time": modified,
                "file_class": classify_file(path),
                "experiment_group": guess_experiment(path),
                "likely_key_output": is_likely_key_output(path),
            })

    return pd.DataFrame(rows), pd.DataFrame(folder_rows)


def build_tree(root: Path, max_depth=5):
    lines = [str(root)]

    def walk(folder: Path, prefix="", depth=0):
        if depth >= max_depth:
            return

        try:
            entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            lines.append(prefix + "[Permission denied]")
            return

        for idx, entry in enumerate(entries):
            connector = "└── " if idx == len(entries) - 1 else "├── "
            rel = entry.name

            if entry.is_dir():
                lines.append(prefix + connector + rel + "/")
                extension = "    " if idx == len(entries) - 1 else "│   "
                walk(entry, prefix + extension, depth + 1)
            else:
                try:
                    size = human_size(entry.stat().st_size)
                except Exception:
                    size = ""
                lines.append(prefix + connector + f"{rel} ({size})")

    walk(root, depth=0)
    return "\n".join(lines)


def summarize_inventory(files_df: pd.DataFrame, folders_df: pd.DataFrame):
    summary = {
        "root": str(ROOT),
        "scan_time": datetime.now().isoformat(timespec="seconds"),
        "total_folders": int(len(folders_df)),
        "total_files": int(len(files_df)),
        "files_by_class": files_df["file_class"].value_counts().to_dict() if not files_df.empty else {},
        "files_by_experiment_group": files_df["experiment_group"].value_counts().to_dict() if not files_df.empty else {},
        "files_by_extension": files_df["extension"].value_counts().to_dict() if not files_df.empty else {},
        "key_outputs_count": int(files_df["likely_key_output"].sum()) if not files_df.empty else 0,
    }
    return summary


def write_recommendation_files(files_df: pd.DataFrame):
    if files_df.empty:
        return

    sec41 = files_df[
        (
            files_df["experiment_group"].isin(["Preprocessing", "LaterCyclePreprocessing", "ModelingReady", "RawData"])
            | files_df["relative_path"].str.lower().str.contains("modeling_ready|pre-processed|processed|audit|summary|missing|logical|distribution", na=False)
        )
        & (
            files_df["file_class"].isin(["table_or_dataset", "report_or_metadata"])
            | files_df["likely_key_output"]
        )
    ].copy()

    sec42 = files_df[
        files_df["experiment_group"].isin(["Experiment1", "Experiment2"])
        & (
            files_df["likely_key_output"]
            | files_df["file_class"].isin(["table_or_dataset", "figure_or_visual", "report_or_metadata"])
        )
    ].copy()

    sec43 = files_df[
        files_df["experiment_group"].isin(["Experiment3", "Experiment4"])
        & (
            files_df["likely_key_output"]
            | files_df["file_class"].isin(["table_or_dataset", "figure_or_visual", "report_or_metadata"])
        )
    ].copy()

    sec44 = files_df[
        files_df["experiment_group"].isin(["Experiment5", "LaterCyclePreprocessing", "ModelingReady"])
        & (
            files_df["likely_key_output"]
            | files_df["file_class"].isin(["table_or_dataset", "figure_or_visual", "report_or_metadata"])
        )
    ].copy()

    sections = {
        "section_4_1_needed_files_dataset_preprocessing.csv": sec41,
        "section_4_2_needed_files_org_learning_explainability.csv": sec42,
        "section_4_3_needed_files_swarm_edge_efficiency.csv": sec43,
        "section_4_4_needed_files_temporal_generalization.csv": sec44,
    }

    cols = [
        "file_name",
        "relative_path",
        "file_class",
        "experiment_group",
        "size_readable",
        "modified_time",
        "likely_key_output",
    ]

    for name, df in sections.items():
        df = df.sort_values(["experiment_group", "file_class", "file_name"]) if not df.empty else df
        df[cols].to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8-sig")


def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Root folder does not exist: {ROOT}")

    print("Scanning experiments folder...")
    print(f"Root: {ROOT}")

    files_df, folders_df = scan_files(ROOT)

    files_df.to_csv(OUTPUT_DIR / "experiments_files_inventory.csv", index=False, encoding="utf-8-sig")
    folders_df.to_csv(OUTPUT_DIR / "experiments_folders_inventory.csv", index=False, encoding="utf-8-sig")

    key_outputs = files_df[files_df["likely_key_output"]].copy() if not files_df.empty else files_df
    if not key_outputs.empty:
        key_outputs = key_outputs.sort_values(["experiment_group", "file_class", "file_name"])
    key_outputs.to_csv(OUTPUT_DIR / "experiments_key_outputs_inventory.csv", index=False, encoding="utf-8-sig")

    tree_text = build_tree(ROOT, max_depth=MAX_TREE_DEPTH)
    (OUTPUT_DIR / "experiments_folder_tree.txt").write_text(tree_text, encoding="utf-8")

    summary = summarize_inventory(files_df, folders_df)
    with open(OUTPUT_DIR / "experiments_inventory_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_recommendation_files(files_df)

    print("\nScan completed successfully.")
    print(f"Output folder: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  {OUTPUT_DIR / 'experiments_files_inventory.csv'}")
    print(f"  {OUTPUT_DIR / 'experiments_folders_inventory.csv'}")
    print(f"  {OUTPUT_DIR / 'experiments_key_outputs_inventory.csv'}")
    print(f"  {OUTPUT_DIR / 'experiments_folder_tree.txt'}")
    print(f"  {OUTPUT_DIR / 'experiments_inventory_summary.json'}")
    print("\nSection-specific upload recommendation files:")
    print(f"  {OUTPUT_DIR / 'section_4_1_needed_files_dataset_preprocessing.csv'}")
    print(f"  {OUTPUT_DIR / 'section_4_2_needed_files_org_learning_explainability.csv'}")
    print(f"  {OUTPUT_DIR / 'section_4_3_needed_files_swarm_edge_efficiency.csv'}")
    print(f"  {OUTPUT_DIR / 'section_4_4_needed_files_temporal_generalization.csv'}")

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

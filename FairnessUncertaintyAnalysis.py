#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Experiment 2 Add-on: Table 10 Fairness, Subgroup Performance, and Uncertainty Analysis
=====================================================================================

Purpose
-------
This script computes the exact numerical values needed for the manuscript table:

Table 10. Demographic subgroup performance and calibration consistency analysis
across the internal NHANES 2011-2018 testing environment.

It uses the same inputs and model flow used in Experiments 1-5:

Inputs
------
D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\modeling_ready\train_2011_2018.csv
D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\modeling_ready\test_2011_2018.csv
D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\Experiment1\Results\experiment1_best_model.pt

Outputs
-------
D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\Experiment2\Results

Generated files
---------------
experiment2_table10_fairness_uncertainty_summary.csv
experiment2_table10_fairness_uncertainty_summary.md
experiment2_subgroup_performance_detailed.csv
experiment2_calibration_bins.csv
experiment2_calibration_curve.png
experiment2_fairness_uncertainty_manifest.json

What it calculates
------------------
1. Subgroup performance by sex, age group, and race/ethnicity when available.
2. Accuracy, balanced accuracy, macro-F1, weighted-F1, MCC, ROC-AUC, and average precision.
3. Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier score.
4. Maximum subgroup performance gaps for key demographic axes.
5. Table-ready values for the manuscript.

Run
---
python experiment2_table10_fairness_uncertainty_analysis.py

Requirements
------------
pip install pandas numpy scikit-learn torch matplotlib
"""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class Config:
    root_dir: str = r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments"
    modeling_ready_subdir: str = "modeling_ready"
    experiment1_results_subdir: str = r"Experiment1\Results"
    experiment2_results_subdir: str = r"Experiment2\Results"

    train_file: str = "train_2011_2018.csv"
    test_file: str = "test_2011_2018.csv"
    checkpoint_file: str = "experiment1_best_model.pt"

    target_candidates: Tuple[str, ...] = (
        "disability_stage",
        "target_binary",
        "merge01_vs_23",
        "functional_risk",
        "target",
        "label",
        "y",
    )

    # Candidate demographic columns. The script automatically uses the first match found.
    sex_candidates: Tuple[str, ...] = (
        "sex", "gender", "RIAGENDR", "riagendr", "Gender", "Sex"
    )
    age_candidates: Tuple[str, ...] = (
        "age", "Age", "RIDAGEYR", "ridageyr", "age_years"
    )
    race_candidates: Tuple[str, ...] = (
        "race_ethnicity", "race", "ethnicity", "RIDRETH1", "RIDRETH3",
        "ridreth1", "ridreth3", "Race", "Ethnicity"
    )

    identifier_columns: Tuple[str, ...] = (
        "SEQN", "seqn", "id", "ID", "participant_id", "Participant_ID",
        "index", "Unnamed: 0",
    )

    batch_size: int = 256
    calibration_bins: int = 10
    min_subgroup_n: int = 30
    random_seed: int = 42
    dpi: int = 300


CFG = Config()
ROOT = Path(CFG.root_dir)
MODELING_DIR = ROOT / CFG.modeling_ready_subdir
EXP1_RESULTS = ROOT / CFG.experiment1_results_subdir
EXP2_RESULTS = ROOT / CFG.experiment2_results_subdir
EXP2_RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = MODELING_DIR / CFG.train_file
TEST_FILE = MODELING_DIR / CFG.test_file
CHECKPOINT_FILE = EXP1_RESULTS / CFG.checkpoint_file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "nutritional_metabolic": [
        "bmi", "hba1c", "glucose", "cholesterol", "hdl", "ldl", "triglyceride",
        "triglycerides", "weight", "height", "metabolic", "diabetes", "fasting",
        "glyco", "lipid",
    ],
    "physiological": [
        "sbp", "dbp", "blood", "pressure", "grip", "strength", "pulse",
        "systolic", "diastolic", "physio",
    ],
    "behavioral": [
        "smoking", "smoke", "met", "activity", "physical", "sedentary",
        "phq", "depression", "exercise", "guideline", "behavior",
    ],
    "demographic_fairness": [
        "age", "sex", "gender", "race", "ethnicity", "education", "income",
        "pir", "poverty", "marital", "household", "demographic",
    ],
    "functional_disability": [
        "difficulty", "adl", "iadl", "mobility", "functional", "disability",
        "arthritis", "polypharmacy", "medication", "limitation", "function",
    ],
}


# =============================================================================
# 2. Utility functions
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found: {path}")


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def detect_target_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    col = first_existing_column(df, candidates)
    if col is None:
        raise ValueError("No target column detected. Checked: " + ", ".join(candidates))
    return col


def get_first_existing(d: dict, keys: Iterable[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def normalize_scalar_dict(x) -> Optional[dict]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, pd.Series):
        return x.to_dict()
    return None


# =============================================================================
# 3. Model classes compatible with Experiment 1
# =============================================================================

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LightweightLearner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SpecialistTeam(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        learners_per_team: int,
        dropout: float,
    ):
        super().__init__()
        self.learners = nn.ModuleList(
            [LightweightLearner(input_dim, hidden_dim, num_classes, dropout) for _ in range(learners_per_team)]
        )
        self.learner_logits = nn.Parameter(torch.zeros(learners_per_team))

    def forward(self, x):
        outputs = torch.stack([learner(x) for learner in self.learners], dim=1)
        weights = torch.softmax(self.learner_logits, dim=0)
        team_output = torch.sum(outputs * weights.view(1, -1, 1), dim=1)
        return team_output, weights


class OrganizationalModel(nn.Module):
    def __init__(
        self,
        domain_indices: Dict[str, List[int]],
        hidden_dim: int,
        num_classes: int,
        learners_per_team: int,
        dropout: float,
    ):
        super().__init__()
        self.domain_indices = domain_indices
        self.team_names = list(domain_indices.keys())
        self.teams = nn.ModuleDict()
        for team_name, idx in domain_indices.items():
            self.teams[team_name] = SpecialistTeam(
                input_dim=len(idx),
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                learners_per_team=learners_per_team,
                dropout=dropout,
            )
        self.team_logits = nn.Parameter(torch.zeros(len(self.team_names)))

    def forward(self, x):
        team_outputs = []
        learner_weights = {}
        for team_name in self.team_names:
            idx = self.domain_indices[team_name]
            out, lw = self.teams[team_name](x[:, idx])
            team_outputs.append(out)
            learner_weights[team_name] = lw
        stacked = torch.stack(team_outputs, dim=1)
        team_weights = torch.softmax(self.team_logits, dim=0)
        logits = torch.sum(stacked * team_weights.view(1, -1, 1), dim=1)
        return logits, team_weights, stacked, learner_weights


# =============================================================================
# 4. Metadata extraction and preprocessing
# =============================================================================

def load_checkpoint(path: Path) -> dict:
    require_file(path)
    checkpoint = torch.load(path, map_location=DEVICE)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Invalid Experiment 1 checkpoint. Expected key: model_state_dict")
    return checkpoint


def infer_hidden_dim_from_state(state: dict) -> Optional[int]:
    for k, v in state.items():
        if k.endswith("net.0.weight"):
            return int(v.shape[0])
    return None


def infer_num_classes_from_state(state: dict) -> Optional[int]:
    for k, v in state.items():
        if k.endswith("net.3.weight"):
            return int(v.shape[0])
    return None


def infer_learners_per_team_from_state(state: dict) -> int:
    learner_ids = set()
    for k in state.keys():
        if ".learners." in k:
            part = k.split(".learners.")[1].split(".")[0]
            if part.isdigit():
                learner_ids.add(int(part))
    return max(learner_ids) + 1 if learner_ids else 3


def infer_team_names_from_state(state: dict) -> List[str]:
    teams = []
    for k in state.keys():
        if k.startswith("teams."):
            parts = k.split(".")
            if len(parts) > 1:
                teams.append(parts[1])
    return sorted(list(set(teams)))


def build_domain_indices_from_map(feature_columns: List[str], domain_map: Dict[str, List[str]]) -> Dict[str, List[int]]:
    col_to_idx = {c: i for i, c in enumerate(feature_columns)}
    out = {}
    for team, cols in domain_map.items():
        idx = [col_to_idx[c] for c in cols if c in col_to_idx]
        if idx:
            out[team] = idx
    if not out:
        raise ValueError("Could not build domain indices from domain_map.")
    return out


def build_domain_indices_from_keywords(feature_columns: List[str], team_names: Optional[List[str]] = None) -> Dict[str, List[int]]:
    lower = [c.lower() for c in feature_columns]
    assigned = set()
    domains = {team: [] for team in (team_names or list(DOMAIN_KEYWORDS.keys()))}

    for domain in domains:
        keywords = DOMAIN_KEYWORDS.get(domain, [])
        for i, c in enumerate(lower):
            if i in assigned:
                continue
            if any(k in c for k in keywords):
                domains[domain].append(i)
                assigned.add(i)

    remaining = [i for i in range(len(feature_columns)) if i not in assigned]
    for domain in list(domains.keys()):
        if not domains[domain] and remaining:
            domains[domain].append(remaining.pop(0))
    for i in remaining:
        smallest = min(domains, key=lambda d: len(domains[d]))
        domains[smallest].append(i)

    return domains


def extract_metadata(checkpoint: dict, train_df: pd.DataFrame) -> dict:
    state = checkpoint["model_state_dict"]
    config = get_first_existing(checkpoint, ["config", "CONFIG", "training_config"], {}) or {}
    meta = get_first_existing(checkpoint, ["metadata", "data_metadata", "checkpoint_manifest"], {}) or {}

    target_col = (
        checkpoint.get("target_column")
        or checkpoint.get("target_col")
        or meta.get("target_column")
        or meta.get("target_col")
        or (config.get("target_column") if isinstance(config, dict) else None)
        or detect_target_column(train_df, CFG.target_candidates)
    )

    feature_columns = (
        checkpoint.get("feature_columns")
        or checkpoint.get("feature_cols")
        or checkpoint.get("features")
        or meta.get("feature_columns")
        or meta.get("feature_cols")
    )

    domain_map = (
        checkpoint.get("domain_map")
        or checkpoint.get("domain_features")
        or checkpoint.get("team_feature_map")
        or meta.get("domain_map")
        or meta.get("domain_features")
    )

    if feature_columns is None and domain_map is not None:
        feature_columns = []
        for _, cols in domain_map.items():
            for c in cols:
                if c not in feature_columns:
                    feature_columns.append(c)

    if feature_columns is None:
        feature_columns = [
            c for c in train_df.columns
            if c != target_col and c not in CFG.identifier_columns and str(c).lower() != "original_split"
        ]

    team_names = (
        checkpoint.get("team_names")
        or checkpoint.get("specialist_team_names")
        or meta.get("team_names")
        or (list(domain_map.keys()) if isinstance(domain_map, dict) else None)
        or infer_team_names_from_state(state)
    )

    if domain_map is not None:
        domain_indices = build_domain_indices_from_map(feature_columns, domain_map)
    else:
        domain_indices = build_domain_indices_from_keywords(feature_columns, team_names)

    class_mapping = (
        checkpoint.get("class_mapping")
        or checkpoint.get("class_to_idx")
        or checkpoint.get("target_mapping")
        or meta.get("class_mapping")
        or meta.get("class_to_idx")
    )

    if class_mapping is None:
        values = sorted(train_df[target_col].dropna().astype(str).unique().tolist())
        class_mapping = {v: i for i, v in enumerate(values)}

    hidden_dim = (
        checkpoint.get("hidden_dim")
        or meta.get("hidden_dim")
        or (config.get("hidden_dim") if isinstance(config, dict) else None)
        or infer_hidden_dim_from_state(state)
        or 16
    )

    num_classes = (
        checkpoint.get("num_classes")
        or meta.get("num_classes")
        or (config.get("num_classes") if isinstance(config, dict) else None)
        or infer_num_classes_from_state(state)
        or len(class_mapping)
    )

    learners_per_team = (
        checkpoint.get("num_learners_per_team")
        or checkpoint.get("n_learners")
        or checkpoint.get("learners_per_team")
        or meta.get("num_learners_per_team")
        or meta.get("n_learners")
        or (config.get("learners_per_team") if isinstance(config, dict) else None)
        or infer_learners_per_team_from_state(state)
    )

    dropout = (
        checkpoint.get("dropout")
        or meta.get("dropout")
        or (config.get("dropout") if isinstance(config, dict) else None)
        or 0.15
    )

    scaler_mean = checkpoint.get("scaler_mean") or checkpoint.get("feature_mean") or meta.get("scaler_mean") or meta.get("feature_mean")
    scaler_scale = checkpoint.get("scaler_scale") or checkpoint.get("feature_scale") or meta.get("scaler_scale") or meta.get("feature_scale")
    train_medians = normalize_scalar_dict(checkpoint.get("train_medians") or meta.get("train_medians"))
    categorical_encoders = checkpoint.get("categorical_encoders") or checkpoint.get("category_maps") or meta.get("categorical_encoders") or {}

    return {
        "target_column": target_col,
        "feature_columns": list(feature_columns),
        "domain_indices": domain_indices,
        "team_names": list(domain_indices.keys()),
        "class_mapping": class_mapping,
        "num_classes": int(num_classes),
        "hidden_dim": int(hidden_dim),
        "learners_per_team": int(learners_per_team),
        "dropout": float(dropout),
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "train_medians": train_medians,
        "categorical_encoders": categorical_encoders,
    }


def drop_identifier_columns(X: pd.DataFrame) -> pd.DataFrame:
    remove = []
    id_set = {x.lower() for x in CFG.identifier_columns}
    for c in X.columns:
        lname = str(c).strip().lower()
        if lname in id_set:
            remove.append(c)
        elif lname.endswith("_id") and X[c].nunique(dropna=True) > 0.90 * len(X):
            remove.append(c)
    return X.drop(columns=remove, errors="ignore")


def fit_feature_encoder(train_X: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    encoders = {}
    for col in train_X.columns:
        if not pd.api.types.is_numeric_dtype(train_X[col]):
            values = pd.Series(train_X[col].astype(str).fillna("MISSING")).unique().tolist()
            encoders[col] = {v: i for i, v in enumerate(values)}
    return encoders


def transform_features(
    X: pd.DataFrame,
    feature_columns: List[str],
    encoders: Dict[str, Dict[str, int]],
    train_medians: pd.Series,
) -> pd.DataFrame:
    out = X.reindex(columns=feature_columns, fill_value=np.nan).copy()
    for col in out.columns:
        if col in encoders:
            out[col] = (
                out[col]
                .astype(str)
                .fillna("MISSING")
                .map(encoders[col])
                .fillna(-1)
                .astype(float)
            )
        elif out[col].dtype == "bool":
            out[col] = out[col].astype(int)
        elif not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(train_medians)
    return out


def prepare_test_data(train_df: pd.DataFrame, test_df: pd.DataFrame, metadata: dict):
    target_col = metadata["target_column"]
    feature_columns = metadata["feature_columns"]
    class_mapping = {str(k): int(v) for k, v in metadata["class_mapping"].items()}

    train_X_raw = drop_identifier_columns(train_df.drop(columns=[target_col], errors="ignore"))
    test_X_raw = drop_identifier_columns(test_df.drop(columns=[target_col], errors="ignore"))

    if metadata.get("categorical_encoders"):
        encoders = metadata["categorical_encoders"]
    else:
        encoders = fit_feature_encoder(train_X_raw.reindex(columns=feature_columns, fill_value=np.nan))

    train_X_num = transform_features(
        train_X_raw,
        feature_columns,
        encoders,
        pd.Series(0.0, index=feature_columns),
    )

    if metadata.get("train_medians") is not None:
        medians = pd.Series(metadata["train_medians"]).reindex(feature_columns).fillna(0)
    else:
        medians = train_X_num.median(numeric_only=True).reindex(feature_columns).fillna(0)

    train_X_num = train_X_num.fillna(medians)
    test_X_num = transform_features(test_X_raw, feature_columns, encoders, medians)

    if metadata.get("scaler_mean") is not None and metadata.get("scaler_scale") is not None:
        mean = np.asarray(metadata["scaler_mean"], dtype=float)
        scale = np.asarray(metadata["scaler_scale"], dtype=float)
        if len(mean) != len(feature_columns):
            raise ValueError("Checkpoint scaler_mean length does not match feature_columns length.")
        scale = np.where(scale == 0, 1.0, scale)
        X_test = (test_X_num.values - mean) / scale
    else:
        scaler = StandardScaler()
        scaler.fit(train_X_num.values)
        X_test = scaler.transform(test_X_num.values)

    y = test_df[target_col].astype(str).map(class_mapping).fillna(-1).astype(int).values
    keep = y >= 0
    return X_test[keep], y[keep], test_df.loc[keep].reset_index(drop=True)


# =============================================================================
# 5. Prediction, metrics, calibration, and subgroup analysis
# =============================================================================

def predict_probabilities(model: nn.Module, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    loader = DataLoader(TabularDataset(X, y), batch_size=CFG.batch_size, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            logits, _, _, _ = model(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def metric_dict(y_true: np.ndarray, probs: np.ndarray, labels: List[int]) -> Dict[str, float]:
    y_pred = probs.argmax(axis=1)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    try:
        if len(labels) == 2:
            out["roc_auc"] = roc_auc_score(y_true, probs[:, 1])
            out["average_precision"] = average_precision_score(y_true, probs[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=labels)
            out["roc_auc"] = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
            out["average_precision"] = average_precision_score(y_bin, probs, average="macro")
    except Exception:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan
    return {k: float(v) if v is not None else np.nan for k, v in out.items()}


def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray, labels: List[int]) -> float:
    y_bin = label_binarize(y_true, classes=labels)
    if len(labels) == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    return float(np.mean(np.sum((probs - y_bin) ** 2, axis=1)))


def calibration_bins(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Tuple[pd.DataFrame, float, float]:
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)

    rows = []
    ece = 0.0
    mce = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i == 0:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf > low) & (conf <= high)
        n = int(mask.sum())
        if n == 0:
            avg_conf = np.nan
            accuracy = np.nan
            gap = np.nan
        else:
            avg_conf = float(conf[mask].mean())
            accuracy = float(correct[mask].mean())
            gap = abs(avg_conf - accuracy)
            ece += (n / len(conf)) * gap
            mce = max(mce, gap)
        rows.append({
            "bin": i + 1,
            "confidence_low": low,
            "confidence_high": high,
            "n": n,
            "avg_confidence": avg_conf,
            "bin_accuracy": accuracy,
            "abs_calibration_gap": gap,
        })

    return pd.DataFrame(rows), float(ece), float(mce)


def standardize_sex(series: pd.Series) -> pd.Series:
    def map_one(v):
        if pd.isna(v):
            return "Unknown"
        s = str(v).strip().lower()
        if s in {"1", "male", "m"}:
            return "Male"
        if s in {"2", "female", "f"}:
            return "Female"
        return str(v)
    return series.apply(map_one)


def age_groups(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        s,
        bins=[-np.inf, 39, 64, np.inf],
        labels=["<40 years", "40-64 years", ">=65 years"],
    ).astype(str).replace("nan", "Unknown")


def subgroup_performance(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    labels: List[int],
) -> Tuple[pd.DataFrame, dict]:
    subgroup_specs = {}
    sex_col = first_existing_column(df_meta, CFG.sex_candidates)
    age_col = first_existing_column(df_meta, CFG.age_candidates)
    race_col = first_existing_column(df_meta, CFG.race_candidates)

    if sex_col:
        subgroup_specs["Sex"] = standardize_sex(df_meta[sex_col])
    if age_col:
        subgroup_specs["Age Group"] = age_groups(df_meta[age_col])
    if race_col:
        subgroup_specs["Race/Ethnicity"] = df_meta[race_col].astype(str).fillna("Unknown")

    rows = []
    for axis, groups in subgroup_specs.items():
        for group in sorted(groups.dropna().unique().tolist()):
            if group == "Unknown":
                continue
            mask = groups == group
            n = int(mask.sum())
            if n < CFG.min_subgroup_n:
                continue
            metrics = metric_dict(y_true[mask.values], probs[mask.values], labels)
            rows.append({
                "axis": axis,
                "subgroup": group,
                "n": n,
                **metrics,
            })

    perf = pd.DataFrame(rows)

    gaps = {}
    for axis in sorted(perf["axis"].unique()) if not perf.empty else []:
        axis_df = perf[perf["axis"] == axis]
        for metric in ["accuracy", "balanced_accuracy", "f1_weighted", "roc_auc"]:
            vals = pd.to_numeric(axis_df[metric], errors="coerce").dropna()
            if len(vals) >= 2:
                gaps[f"{axis}_{metric}_gap"] = float(vals.max() - vals.min())

    return perf, gaps


def plot_calibration(calib: pd.DataFrame, ece: float, path: Path) -> None:
    plot_df = calib.dropna(subset=["avg_confidence", "bin_accuracy"])
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    if not plot_df.empty:
        plt.plot(plot_df["avg_confidence"], plot_df["bin_accuracy"], marker="o", label="Observed")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title(f"Calibration Curve / Reliability Diagram (ECE = {ece:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=CFG.dpi, bbox_inches="tight")
    plt.close()


def build_table10_summary(
    subgroup_df: pd.DataFrame,
    gaps: dict,
    ece: float,
    mce: float,
    brier: float,
    overall: dict,
) -> pd.DataFrame:
    rows = []

    rows.append({
        "Evaluation Category": "Overall internal test performance",
        "Subgroup / Metric": "Accuracy",
        "Value": f"{overall.get('accuracy', np.nan):.4f}",
    })
    rows.append({
        "Evaluation Category": "Overall internal test performance",
        "Subgroup / Metric": "Weighted F1-score",
        "Value": f"{overall.get('f1_weighted', np.nan):.4f}",
    })
    rows.append({
        "Evaluation Category": "Overall internal test performance",
        "Subgroup / Metric": "ROC-AUC",
        "Value": f"{overall.get('roc_auc', np.nan):.4f}",
    })

    if not subgroup_df.empty:
        for axis in ["Sex", "Age Group", "Race/Ethnicity"]:
            axis_df = subgroup_df[subgroup_df["axis"] == axis]
            if axis_df.empty:
                continue
            for _, r in axis_df.iterrows():
                rows.append({
                    "Evaluation Category": axis,
                    "Subgroup / Metric": f"{r['subgroup']} accuracy / weighted F1",
                    "Value": f"{r['accuracy']:.4f} / {r['f1_weighted']:.4f} (n={int(r['n'])})",
                })

    for key, val in gaps.items():
        rows.append({
            "Evaluation Category": "Fairness consistency",
            "Subgroup / Metric": key.replace("_", " "),
            "Value": f"{val:.4f}",
        })

    rows.extend([
        {
            "Evaluation Category": "Calibration",
            "Subgroup / Metric": "Expected Calibration Error (ECE)",
            "Value": f"{ece:.4f}",
        },
        {
            "Evaluation Category": "Calibration",
            "Subgroup / Metric": "Maximum Calibration Error (MCE)",
            "Value": f"{mce:.4f}",
        },
        {
            "Evaluation Category": "Calibration",
            "Subgroup / Metric": "Multiclass Brier score",
            "Value": f"{brier:.4f}",
        },
    ])

    return pd.DataFrame(rows)


# =============================================================================
# 6. Main
# =============================================================================

def main() -> None:
    set_seed(CFG.random_seed)
    print("=" * 80)
    print("Experiment 2 Add-on: Table 10 Fairness and Uncertainty Analysis")
    print("=" * 80)
    print(f"Train file : {TRAIN_FILE}")
    print(f"Test file  : {TEST_FILE}")
    print(f"Checkpoint : {CHECKPOINT_FILE}")
    print(f"Output dir : {EXP2_RESULTS}")
    print(f"Device     : {DEVICE}")

    require_file(TRAIN_FILE)
    require_file(TEST_FILE)
    require_file(CHECKPOINT_FILE)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    train_df.columns = [str(c).strip() for c in train_df.columns]
    test_df.columns = [str(c).strip() for c in test_df.columns]

    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    metadata = extract_metadata(checkpoint, train_df)

    X_test, y_test, test_meta = prepare_test_data(train_df, test_df, metadata)
    labels = list(range(metadata["num_classes"]))

    model = OrganizationalModel(
        domain_indices=metadata["domain_indices"],
        hidden_dim=metadata["hidden_dim"],
        num_classes=metadata["num_classes"],
        learners_per_team=metadata["learners_per_team"],
        dropout=metadata["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    probs = predict_probabilities(model, X_test, y_test)
    overall = metric_dict(y_test, probs, labels)
    subgroup_df, gaps = subgroup_performance(test_meta, y_test, probs, labels)
    calib_df, ece, mce = calibration_bins(y_test, probs, CFG.calibration_bins)
    brier = multiclass_brier_score(y_test, probs, labels)

    table10 = build_table10_summary(subgroup_df, gaps, ece, mce, brier, overall)

    save_dataframe(table10, EXP2_RESULTS / "experiment2_table10_fairness_uncertainty_summary.csv")
    save_dataframe(subgroup_df, EXP2_RESULTS / "experiment2_subgroup_performance_detailed.csv")
    save_dataframe(calib_df, EXP2_RESULTS / "experiment2_calibration_bins.csv")
    plot_calibration(calib_df, ece, EXP2_RESULTS / "experiment2_calibration_curve.png")

    with open(EXP2_RESULTS / "experiment2_table10_fairness_uncertainty_summary.md", "w", encoding="utf-8") as f:
        f.write("# Table 10. Demographic subgroup performance and calibration consistency analysis\n\n")
        f.write(table10.to_markdown(index=False) if hasattr(table10, "to_markdown") else table10.to_csv(index=False))
        f.write("\n")

    manifest = {
        "analysis": "Experiment 2 add-on fairness and uncertainty analysis for Table 10",
        "input_train_file": str(TRAIN_FILE),
        "input_test_file": str(TEST_FILE),
        "checkpoint_file": str(CHECKPOINT_FILE),
        "output_dir": str(EXP2_RESULTS),
        "n_test_records": int(len(y_test)),
        "target_column": metadata["target_column"],
        "num_classes": int(metadata["num_classes"]),
        "feature_count": int(len(metadata["feature_columns"])),
        "team_names": metadata["team_names"],
        "calibration_bins": CFG.calibration_bins,
        "min_subgroup_n": CFG.min_subgroup_n,
        "overall_metrics": overall,
        "ece": ece,
        "mce": mce,
        "brier_score": brier,
        "fairness_gaps": gaps,
        "demographic_columns_detected": {
            "sex": first_existing_column(test_df, CFG.sex_candidates),
            "age": first_existing_column(test_df, CFG.age_candidates),
            "race_ethnicity": first_existing_column(test_df, CFG.race_candidates),
        },
        "device": str(DEVICE),
        "torch_version": torch.__version__,
    }
    save_json(manifest, EXP2_RESULTS / "experiment2_fairness_uncertainty_manifest.json")

    print("\nGenerated manuscript-ready Table 10 values:")
    print(table10.to_string(index=False))
    print("\nSaved outputs to:")
    print(EXP2_RESULTS)
    print("=" * 80)


if __name__ == "__main__":
    main()

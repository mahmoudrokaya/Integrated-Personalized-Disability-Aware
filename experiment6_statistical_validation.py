#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 6: Statistical Validation
====================================

Paper:
Integrated Personalized Disability-Aware Health Intelligence Under Heterogeneous
and Partially Observable Conditions

Purpose
-------
This script adds formal statistical validation for the proposed organizational
learning framework. It is designed to address reproducibility and scientific
rigor requirements by providing:

1. Five-fold stratified cross-validation on the internal NHANES 2011-2018 data.
2. Repeated-seed evaluation.
3. Mean ± standard deviation reporting.
4. 95% confidence intervals.
5. Statistical significance testing between:
   - Static equal cooperation baseline
   - Dynamic Nash-cooperative organizational weighting
6. Optional external temporal evaluation on later-cycle NHANES data using the
   best fold model from each seed.

Input folder
------------
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/modeling_ready

Expected input files
--------------------
train_2011_2018.csv
val_2011_2018.csv
test_2011_2018.csv
external_2017_2023.csv     optional, if available

Output folder
-------------
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/Experiment6/Results

Main outputs
------------
experiment6_cv_all_runs.csv
experiment6_summary_mean_sd_ci.csv
experiment6_significance_tests.csv
experiment6_external_temporal_validation.csv
experiment6_reproducibility_manifest.json
experiment6_statistical_validation_report.md
experiment6_cv_metric_boxplots.png
experiment6_strategy_comparison_barplot.png

Run
---
python experiment6_statistical_validation.py

Requirements
------------
pip install pandas numpy scikit-learn torch matplotlib scipy

Notes
-----
- All preprocessing is fit only on the training fold and then applied unchanged
  to validation/test/external data to prevent leakage.
- Participant identifiers such as SEQN/id columns are excluded from modeling.
- The script is intentionally self-contained and does not overwrite previous
  Experiment 1-5 outputs.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class Experiment6Config:
    root_dir: str = r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments"
    modeling_ready_subdir: str = "modeling_ready"
    output_subdir: str = r"Experiment6\Results"

    train_file: str = "train_2011_2018.csv"
    val_file: str = "val_2011_2018.csv"
    test_file: str = "test_2011_2018.csv"
    external_file: str = "external_2017_2023.csv"

    target_candidates: Tuple[str, ...] = (
        "disability_stage",
        "target_binary",
        "merge01_vs_23",
        "functional_risk",
        "target",
        "label",
        "y",
    )

    identifier_columns: Tuple[str, ...] = (
        "SEQN",
        "seqn",
        "id",
        "ID",
        "participant_id",
        "Participant_ID",
        "index",
        "Unnamed: 0",
    )

    n_splits: int = 5
    repeated_seeds: Tuple[int, ...] = (42, 123, 2024, 2025, 2026)

    # Lightweight model settings aligned with Experiment 1 v2.
    hidden_dim: int = 16
    learners_per_team: int = 3
    dropout: float = 0.15
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    lambda_diversity: float = 0.01
    lambda_entropy: float = 0.005
    lambda_cost: float = 1e-6
    utility_sensitivity: float = 0.10
    min_team_weight: float = 1e-4

    validation_fraction_within_training_fold: float = 0.15
    num_workers: int = 0
    dpi: int = 300


CFG = Experiment6Config()
ROOT_DIR = Path(CFG.root_dir)
MODELING_DIR = ROOT_DIR / CFG.modeling_ready_subdir
OUTPUT_DIR = ROOT_DIR / CFG.output_subdir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# 2. Reproducibility helpers
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


# =============================================================================
# 3. Data handling
# =============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")


def detect_target_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "No target column found. Checked candidates: " + ", ".join(candidates)
    )


def load_internal_dataset() -> pd.DataFrame:
    paths = [
        MODELING_DIR / CFG.train_file,
        MODELING_DIR / CFG.val_file,
        MODELING_DIR / CFG.test_file,
    ]
    for path in paths:
        require_file(path)

    frames = []
    for split_name, path in zip(["train", "validation", "test"], paths):
        df = pd.read_csv(path)
        df["original_split"] = split_name
        frames.append(df)

    data = pd.concat(frames, axis=0, ignore_index=True)
    data.columns = [str(c).strip() for c in data.columns]
    return data


def load_external_dataset_if_available() -> Optional[pd.DataFrame]:
    path = MODELING_DIR / CFG.external_file
    if not path.exists():
        print(f"External file not found. Skipping external evaluation: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def drop_identifier_columns(X: pd.DataFrame) -> pd.DataFrame:
    remove = []
    for c in X.columns:
        name = str(c).strip()
        lname = name.lower()
        if name in CFG.identifier_columns or lname in {x.lower() for x in CFG.identifier_columns}:
            remove.append(c)
        elif lname.endswith("_id") and X[c].nunique(dropna=True) > 0.90 * len(X):
            remove.append(c)
        elif lname == "original_split":
            remove.append(c)
    return X.drop(columns=remove, errors="ignore")


def fit_feature_encoder(train_X: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    encoders: Dict[str, Dict[str, int]] = {}
    for col in train_X.columns:
        if not pd.api.types.is_numeric_dtype(train_X[col]):
            values = pd.Series(train_X[col].astype(str).fillna("MISSING")).unique().tolist()
            encoders[col] = {v: i for i, v in enumerate(values)}
    return encoders


def transform_features(
    X: pd.DataFrame,
    columns: List[str],
    encoders: Dict[str, Dict[str, int]],
    train_medians: Optional[pd.Series] = None,
) -> pd.DataFrame:
    out = X.reindex(columns=columns, fill_value=np.nan).copy()
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
    if train_medians is not None:
        out = out.fillna(train_medians)
    return out


def fit_target_encoder(y: pd.Series) -> Dict[str, int]:
    values = sorted(pd.Series(y).dropna().astype(str).unique().tolist())
    return {v: i for i, v in enumerate(values)}


def transform_target(y: pd.Series, mapping: Dict[str, int]) -> np.ndarray:
    return y.astype(str).map(mapping).fillna(-1).astype(np.int64).values


def build_domain_indices(columns: List[str]) -> Dict[str, List[int]]:
    lower = [c.lower() for c in columns]
    assigned = set()
    domains: Dict[str, List[int]] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        idx = []
        for i, c in enumerate(lower):
            if any(k in c for k in keywords):
                idx.append(i)
                assigned.add(i)
        domains[domain] = idx

    remaining = [i for i in range(len(columns)) if i not in assigned]

    # Guarantee no empty specialist team.
    for domain in list(domains.keys()):
        if not domains[domain] and remaining:
            domains[domain].append(remaining.pop(0))

    for i in remaining:
        smallest = min(domains, key=lambda d: len(domains[d]))
        domains[smallest].append(i)

    empty = [d for d, idx in domains.items() if not idx]
    if empty:
        raise ValueError(f"Specialist domains with no features: {empty}")

    return domains


# =============================================================================
# 4. Dataset and model
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
            [
                LightweightLearner(input_dim, hidden_dim, num_classes, dropout)
                for _ in range(learners_per_team)
            ]
        )
        self.learner_logits = nn.Parameter(torch.zeros(learners_per_team))

    def forward(self, x):
        learner_outputs = torch.stack([learner(x) for learner in self.learners], dim=1)
        learner_weights = torch.softmax(self.learner_logits, dim=0)
        team_output = torch.sum(learner_outputs * learner_weights.view(1, -1, 1), dim=1)
        return team_output, learner_weights


class OrganizationalModel(nn.Module):
    def __init__(
        self,
        domain_indices: Dict[str, List[int]],
        hidden_dim: int,
        num_classes: int,
        learners_per_team: int,
        dropout: float,
        strategy: str = "dynamic_nash",
    ):
        super().__init__()
        if strategy not in {"dynamic_nash", "static_equal"}:
            raise ValueError("strategy must be 'dynamic_nash' or 'static_equal'")

        self.domain_indices = domain_indices
        self.team_names = list(domain_indices.keys())
        self.strategy = strategy

        self.teams = nn.ModuleDict()
        for team_name, indices in domain_indices.items():
            self.teams[team_name] = SpecialistTeam(
                input_dim=len(indices),
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                learners_per_team=learners_per_team,
                dropout=dropout,
            )

        if strategy == "dynamic_nash":
            self.team_logits = nn.Parameter(torch.zeros(len(self.team_names)))
        else:
            self.register_buffer("static_weights", torch.ones(len(self.team_names)) / len(self.team_names))

    def forward(self, x):
        team_outputs = []
        learner_weights = {}
        for team_name in self.team_names:
            idx = self.domain_indices[team_name]
            out, lw = self.teams[team_name](x[:, idx])
            team_outputs.append(out)
            learner_weights[team_name] = lw

        stacked = torch.stack(team_outputs, dim=1)
        if self.strategy == "dynamic_nash":
            team_weights = torch.softmax(self.team_logits, dim=0)
        else:
            team_weights = self.static_weights

        logits = torch.sum(stacked * team_weights.view(1, -1, 1), dim=1)
        return logits, team_weights, stacked, learner_weights


def diversity_regularization(team_outputs: torch.Tensor) -> torch.Tensor:
    """Encourage specialist teams not to collapse into identical logits."""
    if team_outputs.shape[1] <= 1:
        return torch.tensor(0.0, device=team_outputs.device)
    probs = torch.softmax(team_outputs, dim=-1)
    sims = []
    for i in range(probs.shape[1]):
        for j in range(i + 1, probs.shape[1]):
            sims.append(F.cosine_similarity(probs[:, i, :], probs[:, j, :], dim=1).mean())
    return torch.stack(sims).mean() if sims else torch.tensor(0.0, device=team_outputs.device)


def entropy_regularization(team_weights: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    entropy = -torch.sum(team_weights * torch.log(team_weights + eps))
    return -entropy


def cost_regularization(model: nn.Module) -> torch.Tensor:
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return torch.tensor(float(n_params), device=DEVICE)


# =============================================================================
# 5. Training and evaluation
# =============================================================================

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, labels: List[int]) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    out = {
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "precision_macro": precision_score(y_true, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, preds, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, preds),
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


def evaluate_model(model: nn.Module, loader: DataLoader, labels: List[int]) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits, _, _, _ = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_y.append(yb.numpy())

    probs = np.vstack(all_probs)
    y_true = np.concatenate(all_y)
    metrics = compute_metrics(y_true, probs, labels)
    return metrics, y_true, probs


def train_one_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    domain_indices: Dict[str, List[int]],
    num_classes: int,
    seed: int,
    strategy: str,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    set_seed(seed)

    labels = list(range(num_classes))
    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    model = OrganizationalModel(
        domain_indices=domain_indices,
        hidden_dim=CFG.hidden_dim,
        num_classes=num_classes,
        learners_per_team=CFG.learners_per_team,
        dropout=CFG.dropout,
        strategy=strategy,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)

    best_state = None
    best_val_f1 = -np.inf
    patience_counter = 0
    history = []

    for epoch in range(1, CFG.max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits, team_weights, team_outputs, _ = model(xb)
            ce_loss = F.cross_entropy(logits, yb)
            div_loss = diversity_regularization(team_outputs)
            ent_loss = entropy_regularization(team_weights)
            cst_loss = cost_regularization(model)

            loss = (
                ce_loss
                + CFG.lambda_diversity * div_loss
                + CFG.lambda_entropy * ent_loss
                + CFG.lambda_cost * cst_loss
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics, _, _ = evaluate_model(model, val_loader, labels)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else np.nan,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        val_f1 = val_metrics.get("f1_weighted", -np.inf)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= CFG.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def prepare_fold_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    seed: int,
):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Drop target missing.
    train_df = train_df[train_df[target_col].notna()].reset_index(drop=True)
    test_df = test_df[test_df[target_col].notna()].reset_index(drop=True)

    y_map = fit_target_encoder(train_df[target_col])
    y_train_full = transform_target(train_df[target_col], y_map)
    y_test = transform_target(test_df[target_col], y_map)

    # Remove target classes in test not seen during training.
    test_keep = y_test >= 0
    test_df = test_df.loc[test_keep].reset_index(drop=True)
    y_test = y_test[test_keep]

    X_train_raw = drop_identifier_columns(train_df.drop(columns=[target_col], errors="ignore"))
    X_test_raw = drop_identifier_columns(test_df.drop(columns=[target_col], errors="ignore"))

    feature_columns = list(X_train_raw.columns)
    encoders = fit_feature_encoder(X_train_raw)
    X_train_num = transform_features(X_train_raw, feature_columns, encoders)
    medians = X_train_num.median(numeric_only=True).fillna(0)
    X_train_num = X_train_num.fillna(medians)
    X_test_num = transform_features(X_test_raw, feature_columns, encoders, medians)

    # Train-fold-only scaling.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num.values)
    X_test_scaled = scaler.transform(X_test_num.values)

    domain_indices = build_domain_indices(feature_columns)

    # Internal validation subset from the fold-training portion only.
    stratify = y_train_full if len(np.unique(y_train_full)) > 1 else None
    tr_idx, val_idx = train_test_split(
        np.arange(len(y_train_full)),
        test_size=CFG.validation_fraction_within_training_fold,
        random_state=seed,
        stratify=stratify,
    )

    return {
        "X_train": X_train_scaled[tr_idx],
        "y_train": y_train_full[tr_idx],
        "X_val": X_train_scaled[val_idx],
        "y_val": y_train_full[val_idx],
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "target_mapping": y_map,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "train_medians": medians.to_dict(),
        "encoders": encoders,
        "domain_indices": domain_indices,
        "num_classes": len(y_map),
    }


def prepare_external_with_fold_preprocessing(
    external_df: pd.DataFrame,
    target_col: str,
    fold_info: dict,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if external_df is None:
        return None

    ext_target_candidates = [
        target_col,
        "target_binary",
        "disability_stage",
        "fallback_function_risk",
        "self_rated_health_risk",
        "functional_risk",
        "target",
        "label",
        "y",
    ]
    ext_target = None
    for col in ext_target_candidates:
        if col in external_df.columns:
            ext_target = col
            break
    if ext_target is None:
        return None

    df = external_df.copy()
    df = df[df[ext_target].notna()].reset_index(drop=True)
    if df.empty:
        return None

    y_ext = transform_target(df[ext_target], fold_info["target_mapping"])
    keep = y_ext >= 0
    if keep.sum() == 0:
        # External labels may be binary while internal labels are multiclass.
        # In this case, formal same-label evaluation is not valid.
        return None

    df = df.loc[keep].reset_index(drop=True)
    y_ext = y_ext[keep]

    X_ext_raw = drop_identifier_columns(df.drop(columns=[ext_target], errors="ignore"))
    X_ext_num = transform_features(
        X_ext_raw,
        fold_info["feature_columns"],
        fold_info["encoders"],
        pd.Series(fold_info["train_medians"]),
    )
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(fold_info["scaler_mean"], dtype=float)
    scaler.scale_ = np.asarray(fold_info["scaler_scale"], dtype=float)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(fold_info["feature_columns"])
    X_ext = scaler.transform(X_ext_num.values)
    return X_ext, y_ext


# =============================================================================
# 6. Statistical summaries
# =============================================================================

def mean_sd_ci(values: pd.Series, confidence: float = 0.95) -> Dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().values.astype(float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if n > 1:
        if SCIPY_AVAILABLE:
            tcrit = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
        else:
            tcrit = 1.96
        half_width = tcrit * sd / math.sqrt(n)
    else:
        half_width = 0.0
    return {
        "n": int(n),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def build_summary_table(all_runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "mcc",
        "roc_auc",
        "average_precision",
    ]
    rows = []
    for dataset_name in sorted(all_runs["dataset"].unique()):
        for strategy in sorted(all_runs["strategy"].unique()):
            sub = all_runs[(all_runs["dataset"] == dataset_name) & (all_runs["strategy"] == strategy)]
            if sub.empty:
                continue
            for metric in metrics:
                if metric not in sub.columns:
                    continue
                stats_row = mean_sd_ci(sub[metric])
                rows.append({
                    "dataset": dataset_name,
                    "strategy": strategy,
                    "metric": metric,
                    **stats_row,
                    "mean_sd": f"{stats_row['mean']:.4f} ± {stats_row['sd']:.4f}" if not np.isnan(stats_row["mean"]) else "NA",
                    "ci95": f"[{stats_row['ci95_low']:.4f}, {stats_row['ci95_high']:.4f}]" if not np.isnan(stats_row["ci95_low"]) else "NA",
                })
    return pd.DataFrame(rows)


def significance_tests(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Paired tests between static_equal and dynamic_nash using seed/fold pairs."""
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "mcc",
        "roc_auc",
        "average_precision",
    ]
    rows = []
    for dataset_name in sorted(all_runs["dataset"].unique()):
        dataset_df = all_runs[all_runs["dataset"] == dataset_name].copy()
        for metric in metrics:
            if metric not in dataset_df.columns:
                continue
            wide = dataset_df.pivot_table(
                index=["seed", "fold"],
                columns="strategy",
                values=metric,
                aggfunc="mean",
            ).dropna()
            if not {"static_equal", "dynamic_nash"}.issubset(wide.columns):
                continue
            a = wide["static_equal"].astype(float).values
            b = wide["dynamic_nash"].astype(float).values
            diff = b - a

            row = {
                "dataset": dataset_name,
                "comparison": "dynamic_nash_minus_static_equal",
                "metric": metric,
                "n_pairs": int(len(diff)),
                "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                "sd_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
            }

            if SCIPY_AVAILABLE and len(diff) > 1:
                try:
                    t_stat, p_t = stats.ttest_rel(b, a, nan_policy="omit")
                except Exception:
                    t_stat, p_t = np.nan, np.nan
                try:
                    w_stat, p_w = stats.wilcoxon(b, a, zero_method="wilcox")
                except Exception:
                    w_stat, p_w = np.nan, np.nan
                row.update({
                    "paired_t_statistic": float(t_stat) if t_stat is not None else np.nan,
                    "paired_t_p_value": float(p_t) if p_t is not None else np.nan,
                    "wilcoxon_statistic": float(w_stat) if w_stat is not None else np.nan,
                    "wilcoxon_p_value": float(p_w) if p_w is not None else np.nan,
                    "significant_p_lt_0_05_ttest": bool(p_t < 0.05) if not np.isnan(p_t) else False,
                    "significant_p_lt_0_05_wilcoxon": bool(p_w < 0.05) if not np.isnan(p_w) else False,
                })
            else:
                row.update({
                    "paired_t_statistic": np.nan,
                    "paired_t_p_value": np.nan,
                    "wilcoxon_statistic": np.nan,
                    "wilcoxon_p_value": np.nan,
                    "significant_p_lt_0_05_ttest": False,
                    "significant_p_lt_0_05_wilcoxon": False,
                })
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 7. Figures and markdown report
# =============================================================================

def make_figures(all_runs: pd.DataFrame, summary: pd.DataFrame) -> None:
    key_metrics = ["accuracy", "balanced_accuracy", "f1_weighted", "roc_auc"]
    internal = all_runs[all_runs["dataset"] == "internal_cv"].copy()

    for metric in key_metrics:
        if metric not in internal.columns:
            continue
        plt.figure(figsize=(8, 6))
        data = [
            internal[internal["strategy"] == s][metric].dropna().values
            for s in ["static_equal", "dynamic_nash"]
        ]
        plt.boxplot(data, labels=["Static Equal", "Dynamic Nash"])
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Experiment 6: Cross-Validated {metric.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"experiment6_cv_{metric}_boxplot.png", dpi=CFG.dpi, bbox_inches="tight")
        plt.close()

    # Summary bar plot for weighted F1.
    metric = "f1_weighted"
    rows = summary[(summary["dataset"] == "internal_cv") & (summary["metric"] == metric)].copy()
    if not rows.empty:
        rows = rows.sort_values("strategy")
        plt.figure(figsize=(8, 6))
        x = np.arange(len(rows))
        means = rows["mean"].values
        errs = means - rows["ci95_low"].values
        plt.bar(x, means, yerr=errs, capsize=5)
        plt.xticks(x, rows["strategy"].str.replace("_", " ").str.title(), rotation=10)
        plt.ylabel("Weighted F1-score")
        plt.title("Experiment 6: Mean Weighted F1 with 95% CI")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "experiment6_strategy_comparison_barplot.png", dpi=CFG.dpi, bbox_inches="tight")
        plt.close()


def write_markdown_report(summary: pd.DataFrame, sig: pd.DataFrame, manifest: dict) -> None:
    path = OUTPUT_DIR / "experiment6_statistical_validation_report.md"

    def table_or_note(df: pd.DataFrame, max_rows: int = 30):
        if df.empty:
            return "No records available."
        return df.head(max_rows).to_markdown(index=False)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Experiment 6: Statistical Validation\n\n")
        f.write("## Purpose\n\n")
        f.write(
            "This experiment provides formal statistical validation using five-fold "
            "stratified cross-validation, repeated seeds, mean ± standard deviation, "
            "95% confidence intervals, and paired statistical testing.\n\n"
        )
        f.write("## Reproducibility Configuration\n\n")
        f.write("```json\n")
        f.write(json.dumps(manifest, indent=2))
        f.write("\n```\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(table_or_note(summary))
        f.write("\n\n")
        f.write("## Significance Tests\n\n")
        f.write(table_or_note(sig))
        f.write("\n\n")
        f.write("## Interpretation Guidance\n\n")
        f.write(
            "The internal cross-validation table should be used to report mean ± SD and "
            "95% confidence intervals in the manuscript. The significance-test table "
            "provides paired comparisons between static equal cooperation and dynamic "
            "Nash-cooperative weighting across matched seed-fold runs.\n"
        )


# =============================================================================
# 8. Main experiment
# =============================================================================

def run_experiment6() -> None:
    start_time = time.time()
    print("=" * 80)
    print("Experiment 6: Statistical Validation")
    print("=" * 80)
    print(f"Input folder : {MODELING_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Device       : {DEVICE}")

    internal_df = load_internal_dataset()
    external_df = load_external_dataset_if_available()
    target_col = detect_target_column(internal_df, CFG.target_candidates)

    internal_df = internal_df[internal_df[target_col].notna()].reset_index(drop=True)
    y_all_raw = internal_df[target_col].astype(str).values

    all_run_rows = []
    all_history_rows = []
    external_rows = []

    for seed in CFG.repeated_seeds:
        set_seed(seed)
        skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=seed)

        for fold_id, (train_idx, test_idx) in enumerate(skf.split(internal_df, y_all_raw), start=1):
            print(f"\nSeed {seed} | Fold {fold_id}/{CFG.n_splits}")
            train_fold_df = internal_df.iloc[train_idx].reset_index(drop=True)
            test_fold_df = internal_df.iloc[test_idx].reset_index(drop=True)

            fold_info = prepare_fold_data(train_fold_df, test_fold_df, target_col, seed + fold_id)
            labels = list(range(fold_info["num_classes"]))

            test_loader = DataLoader(
                TabularDataset(fold_info["X_test"], fold_info["y_test"]),
                batch_size=CFG.batch_size,
                shuffle=False,
                num_workers=CFG.num_workers,
            )

            for strategy in ["static_equal", "dynamic_nash"]:
                print(f"  Training strategy: {strategy}")
                model, history = train_one_model(
                    fold_info["X_train"],
                    fold_info["y_train"],
                    fold_info["X_val"],
                    fold_info["y_val"],
                    fold_info["domain_indices"],
                    fold_info["num_classes"],
                    seed=seed + fold_id,
                    strategy=strategy,
                )
                metrics, y_true, probs = evaluate_model(model, test_loader, labels)
                row = {
                    "dataset": "internal_cv",
                    "seed": seed,
                    "fold": fold_id,
                    "strategy": strategy,
                    "n_train": int(len(fold_info["y_train"])),
                    "n_validation": int(len(fold_info["y_val"])),
                    "n_test": int(len(fold_info["y_test"])),
                    **metrics,
                }
                all_run_rows.append(row)

                for h in history:
                    all_history_rows.append({
                        "seed": seed,
                        "fold": fold_id,
                        "strategy": strategy,
                        **h,
                    })

                # Optional external evaluation using same trained model and fold preprocessing.
                ext_prepared = prepare_external_with_fold_preprocessing(external_df, target_col, fold_info)
                if ext_prepared is not None:
                    X_ext, y_ext = ext_prepared
                    ext_loader = DataLoader(
                        TabularDataset(X_ext, y_ext),
                        batch_size=CFG.batch_size,
                        shuffle=False,
                        num_workers=CFG.num_workers,
                    )
                    ext_metrics, _, _ = evaluate_model(model, ext_loader, labels)
                    external_rows.append({
                        "dataset": "external_temporal",
                        "seed": seed,
                        "fold": fold_id,
                        "strategy": strategy,
                        "n_external": int(len(y_ext)),
                        **ext_metrics,
                    })

    all_runs = pd.DataFrame(all_run_rows)
    history_df = pd.DataFrame(all_history_rows)
    external_eval = pd.DataFrame(external_rows)

    if not external_eval.empty:
        combined_for_summary = pd.concat([all_runs, external_eval], ignore_index=True, sort=False)
    else:
        combined_for_summary = all_runs.copy()

    summary = build_summary_table(combined_for_summary)
    sig = significance_tests(combined_for_summary)

    save_dataframe(all_runs, OUTPUT_DIR / "experiment6_cv_all_runs.csv")
    save_dataframe(history_df, OUTPUT_DIR / "experiment6_training_history_all_runs.csv")
    save_dataframe(summary, OUTPUT_DIR / "experiment6_summary_mean_sd_ci.csv")
    save_dataframe(sig, OUTPUT_DIR / "experiment6_significance_tests.csv")
    if not external_eval.empty:
        save_dataframe(external_eval, OUTPUT_DIR / "experiment6_external_temporal_validation.csv")

    manifest = {
        "experiment": "Experiment 6 - Statistical Validation",
        "input_folder": str(MODELING_DIR),
        "output_folder": str(OUTPUT_DIR),
        "input_files": {
            "train": CFG.train_file,
            "validation": CFG.val_file,
            "test": CFG.test_file,
            "external_optional": CFG.external_file,
        },
        "target_column_detected": target_col,
        "n_internal_records": int(len(internal_df)),
        "external_available": external_df is not None,
        "n_external_records": int(len(external_df)) if external_df is not None else 0,
        "n_splits": CFG.n_splits,
        "repeated_seeds": list(CFG.repeated_seeds),
        "strategies": ["static_equal", "dynamic_nash"],
        "model_settings": asdict(CFG),
        "device": str(DEVICE),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "scipy_available": SCIPY_AVAILABLE,
        "runtime_seconds": round(time.time() - start_time, 2),
    }
    save_json(manifest, OUTPUT_DIR / "experiment6_reproducibility_manifest.json")

    make_figures(combined_for_summary, summary)
    write_markdown_report(summary, sig, manifest)

    print("\n" + "=" * 80)
    print("Experiment 6 completed successfully.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment6()

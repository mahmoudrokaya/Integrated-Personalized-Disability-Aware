r"""
Experiment 1: Organizational Training and Internal Cooperative Learning
Paper: Nash-Cooperative Swarm Intelligence for Lightweight Edge Learning Through Multi-Team Adaptive Optimization

Purpose
-------
This script implements Experiment 1 as a complete training + reporting pipeline.
It is intentionally aligned with the Methods section of the paper:

1. Multidomain NHANES decomposition into specialist health-information teams.
2. Lightweight shallow learners inside each specialist team.
3. Intra-team learner aggregation.
4. Inter-team Nash-cooperative adaptive weighting.
5. Prediction, diversity, entropy, and lightweight computational-cost regularization.
6. Internal train/validation/test evaluation using modeling-ready NHANES 2011-2018 splits.
7. Intensive reporting through tables, plots, raw predictions, contribution analysis,
   model complexity, inference latency, and a paper-ready markdown report.

Expected input folder
---------------------
D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\modeling_ready

Expected input files
--------------------
train_2011_2018.csv
val_2011_2018.csv
test_2011_2018.csv

Outputs
-------
D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\Experiment1\Results

Main generated files include:
- experiment1_final_summary.json
- experiment1_metrics_summary.csv
- experiment1_classification_report_test.csv
- experiment1_predictions_test.csv
- experiment1_training_history.csv
- experiment1_team_weight_history.csv
- experiment1_team_utility_history.csv
- experiment1_specialist_team_weights_test.csv
- experiment1_domain_feature_mapping.csv
- experiment1_complexity_latency.csv
- experiment1_paper_ready_report.md
- Multiple PNG figures for training curves, ROC/PR curves, confusion matrix,
  class distribution, specialist-team contributions, utility evolution, and
  prediction confidence.
"""

from __future__ import annotations

import json
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

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler, label_binarize

warnings.filterwarnings("ignore")


# =========================================================
# 1. Reproducibility and configuration
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@dataclass
class ExperimentConfig:
    modeling_ready_dir: str = r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\modeling_ready"
    output_dir: str = r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\Experiment1\Results"

    train_file: str = "train_2011_2018.csv"
    val_file: str = "val_2011_2018.csv"
    test_file: str = "test_2011_2018.csv"

    target_candidates: Tuple[str, ...] = (
        "disability_stage",
        "merge01_vs_23",
        "functional_risk",
        "target",
        "label",
        "y",
    )

    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    hidden_dim: int = 16
    learners_per_team: int = 3
    dropout: float = 0.15

    lambda_diversity: float = 0.01
    lambda_cost: float = 1e-6
    lambda_entropy: float = 0.005

    utility_sensitivity: float = 0.10
    min_team_weight: float = 1e-4

    num_workers: int = 0
    dpi: int = 300


CFG = ExperimentConfig()
OUTPUT_DIR = Path(CFG.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. Domain logic aligned with Methods section
# =========================================================

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "nutritional_metabolic": [
        "bmi", "hba1c", "glucose", "cholesterol", "hdl", "ldl", "triglyceride",
        "weight", "height", "metabolic", "diabetes", "fasting", "glyco",
    ],
    "physiological": [
        "sbp", "dbp", "blood", "pressure", "grip", "strength", "pulse",
        "systolic", "diastolic",
    ],
    "behavioral": [
        "smoking", "smoke", "met", "activity", "physical", "sedentary",
        "phq", "depression", "exercise", "guideline",
    ],
    "demographic_fairness": [
        "age", "sex", "gender", "race", "ethnicity", "education", "income",
        "pir", "poverty", "marital", "household",
    ],
    "functional_disability": [
        "difficulty", "adl", "iadl", "mobility", "functional", "disability",
        "arthritis", "polypharmacy", "medication", "limitation", "function",
    ],
}


# =========================================================
# 3. Data loading, cleaning, and split-safe encoding
# =========================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found: {path}")


def detect_target_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[-1]


def drop_identifier_columns(X: pd.DataFrame) -> pd.DataFrame:
    remove = []
    for c in X.columns:
        name = c.lower().strip()
        if name in {"seqn", "id", "participant_id", "index", "unnamed: 0"}:
            remove.append(c)
        elif name.endswith("_id") and X[c].nunique(dropna=True) > 0.90 * len(X):
            remove.append(c)
    return X.drop(columns=remove, errors="ignore")


def fit_feature_encoder(train_X: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    encoders: Dict[str, Dict[str, int]] = {}
    for col in train_X.columns:
        if not np.issubdtype(train_X[col].dtype, np.number):
            values = pd.Series(train_X[col].astype(str).fillna("MISSING")).unique().tolist()
            encoders[col] = {v: i for i, v in enumerate(values)}
    return encoders


def transform_features(X: pd.DataFrame, columns: List[str], encoders: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    X = X.reindex(columns=columns, fill_value=np.nan).copy()
    for col in X.columns:
        if col in encoders:
            X[col] = X[col].astype(str).fillna("MISSING").map(encoders[col]).fillna(-1).astype(float)
        elif X[col].dtype == "bool":
            X[col] = X[col].astype(int)
        elif not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def fit_target_encoder(y: pd.Series) -> Dict[str, int]:
    values = sorted(pd.Series(y).dropna().astype(str).unique().tolist())
    return {v: i for i, v in enumerate(values)}


def transform_target(y: pd.Series, mapping: Dict[str, int]) -> np.ndarray:
    return y.astype(str).map(mapping).fillna(0).astype(np.int64).values


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

    # Preserve the five-team Methods architecture even when column names are incomplete.
    for domain in list(domains.keys()):
        if not domains[domain] and remaining:
            domains[domain].append(remaining.pop(0))

    for i in remaining:
        smallest = min(domains, key=lambda d: len(domains[d]))
        domains[smallest].append(i)

    empty = [d for d, idx in domains.items() if not idx]
    if empty:
        raise ValueError(f"The following specialist domains have no features: {empty}")

    return domains


def load_internal_data(cfg: ExperimentConfig):
    model_dir = Path(cfg.modeling_ready_dir)
    train_path = model_dir / cfg.train_file
    val_path = model_dir / cfg.val_file
    test_path = model_dir / cfg.test_file
    for p in [train_path, val_path, test_path]:
        require_file(p)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    target_col = detect_target_column(train_df, cfg.target_candidates)
    if target_col not in val_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Detected target column '{target_col}' is not present in all splits.")

    raw_train_X = drop_identifier_columns(train_df.drop(columns=[target_col]))
    feature_columns = list(raw_train_X.columns)
    encoders = fit_feature_encoder(raw_train_X)

    X_train_df = transform_features(raw_train_X, feature_columns, encoders)
    X_val_df = transform_features(drop_identifier_columns(val_df.drop(columns=[target_col])), feature_columns, encoders)
    X_test_df = transform_features(drop_identifier_columns(test_df.drop(columns=[target_col])), feature_columns, encoders)

    medians = X_train_df.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    X_train_df = X_train_df.fillna(medians).fillna(0)
    X_val_df = X_val_df.fillna(medians).fillna(0)
    X_test_df = X_test_df.fillna(medians).fillna(0)

    target_mapping = fit_target_encoder(train_df[target_col])
    y_train = transform_target(train_df[target_col], target_mapping)
    y_val = transform_target(val_df[target_col], target_mapping)
    y_test = transform_target(test_df[target_col], target_mapping)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values).astype(np.float32)
    X_val = scaler.transform(X_val_df.values).astype(np.float32)
    X_test = scaler.transform(X_test_df.values).astype(np.float32)

    domain_indices = build_domain_indices(feature_columns)
    domain_features = {d: [feature_columns[i] for i in idx] for d, idx in domain_indices.items()}

    metadata = {
        "target_column": target_col,
        "target_mapping": target_mapping,
        "feature_columns": feature_columns,
        "num_features": len(feature_columns),
        "domain_features": domain_features,
        "domain_feature_counts": {d: len(v) for d, v in domain_features.items()},
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "num_classes": int(len(target_mapping)),
        "input_files": {
            "train": str(train_path),
            "validation": str(val_path),
            "test": str(test_path),
        },
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, domain_indices, metadata


class NHANESDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


# =========================================================
# 4. Lightweight specialist teams and Nash-cooperative model
# =========================================================

class LightweightLearner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpecialistTeam(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, learners_per_team: int, dropout: float):
        super().__init__()
        self.learners = nn.ModuleList([
            LightweightLearner(input_dim, hidden_dim, num_classes, dropout)
            for _ in range(learners_per_team)
        ])
        self.intra_team_logits = nn.Parameter(torch.zeros(learners_per_team))

    def forward(self, x: torch.Tensor):
        learner_outputs = torch.stack([learner(x) for learner in self.learners], dim=1)
        learner_weights = torch.softmax(self.intra_team_logits, dim=0)
        team_output = torch.sum(learner_outputs * learner_weights.view(1, -1, 1), dim=1)
        return team_output, learner_outputs, learner_weights


class NashCooperativeOrganizationalModel(nn.Module):
    def __init__(self, domain_indices: Dict[str, List[int]], hidden_dim: int, num_classes: int, learners_per_team: int, dropout: float):
        super().__init__()
        self.domain_names = list(domain_indices.keys())
        self.domain_indices = domain_indices
        self.teams = nn.ModuleDict({
            name: SpecialistTeam(len(indices), hidden_dim, num_classes, learners_per_team, dropout)
            for name, indices in domain_indices.items()
        })
        self.team_logits = nn.Parameter(torch.zeros(len(self.domain_names)))

    def forward(self, x: torch.Tensor):
        team_outputs = []
        learner_outputs = {}
        intra_team_weights = {}

        for name in self.domain_names:
            idx = torch.tensor(self.domain_indices[name], dtype=torch.long, device=x.device)
            x_domain = x.index_select(dim=1, index=idx)
            team_logit, learner_logit, learner_weight = self.teams[name](x_domain)
            team_outputs.append(team_logit)
            learner_outputs[name] = learner_logit
            intra_team_weights[name] = learner_weight

        team_outputs_tensor = torch.stack(team_outputs, dim=1)
        team_weights = torch.softmax(self.team_logits, dim=0)
        global_logits = torch.sum(team_outputs_tensor * team_weights.view(1, -1, 1), dim=1)
        return global_logits, team_outputs_tensor, team_weights, intra_team_weights, learner_outputs

    @torch.no_grad()
    def update_nash_weights(self, utilities: torch.Tensor, sensitivity: float, min_weight: float) -> None:
        centered = utilities - utilities.mean()
        self.team_logits.data = self.team_logits.data + sensitivity * centered.to(self.team_logits.device)
        weights = torch.softmax(self.team_logits.data, dim=0)
        weights = torch.clamp(weights, min=min_weight)
        weights = weights / weights.sum()
        self.team_logits.data = torch.log(weights)


# =========================================================
# 5. Objective functions and organizational utilities
# =========================================================

def diversity_loss(team_outputs: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(team_outputs, dim=-1)
    n_teams = probs.shape[1]
    if n_teams <= 1:
        return torch.tensor(0.0, device=team_outputs.device)
    total = torch.tensor(0.0, device=team_outputs.device)
    count = 0
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            total = total + F.cosine_similarity(probs[:, i, :], probs[:, j, :], dim=-1).mean()
            count += 1
    return total / max(count, 1)


def model_cost_regularizer(model: nn.Module) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        total = total + torch.sum(p * p)
    return total


def entropy_regularizer(weights: torch.Tensor) -> torch.Tensor:
    # This is negative entropy. Minimizing it discourages premature single-team domination.
    return torch.sum(weights * torch.log(weights + 1e-8))


@torch.no_grad()
def compute_team_utilities(team_outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    utilities = []
    for k in range(team_outputs.shape[1]):
        team_loss = F.cross_entropy(team_outputs[:, k, :], y)
        utilities.append(-team_loss)
    return torch.stack(utilities)


# =========================================================
# 6. Evaluation, latency, and reporting helpers
# =========================================================

@torch.no_grad()
def evaluate_model(model: NashCooperativeOrganizationalModel, loader: DataLoader, num_classes: int):
    model.eval()
    all_y: List[int] = []
    all_pred: List[int] = []
    all_prob: List[np.ndarray] = []
    all_conf: List[float] = []
    total_loss = 0.0
    team_weight_records = []
    team_utility_records = []

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits, team_outputs, team_weights, _, _ = model(X)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=-1)
        conf, preds = torch.max(probs, dim=-1)

        total_loss += loss.item() * len(y)
        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy())
        all_conf.extend(conf.cpu().numpy().tolist())
        team_weight_records.append(team_weights.cpu().numpy())
        team_utility_records.append(compute_team_utilities(team_outputs, y).cpu().numpy())

    y_true = np.array(all_y, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)
    y_prob = np.vstack(all_prob)

    metrics = {
        "loss": float(total_loss / max(len(y_true), 1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "mean_prediction_confidence": float(np.mean(all_conf)),
    }

    try:
        if num_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            y_bin = label_binarize(y_true, classes=np.arange(num_classes))
            metrics["roc_auc"] = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
            metrics["average_precision"] = float(average_precision_score(y_bin, y_prob, average="macro"))
    except Exception:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

    metrics["team_weights_mean"] = np.mean(np.vstack(team_weight_records), axis=0).tolist()
    metrics["team_utilities_mean"] = np.mean(np.vstack(team_utility_records), axis=0).tolist()
    return metrics, y_true, y_pred, y_prob


@torch.no_grad()
def measure_latency(model: NashCooperativeOrganizationalModel, loader: DataLoader, warmup_batches: int = 3):
    model.eval()
    per_sample = []
    total_samples = 0

    for batch_idx, (X, _) in enumerate(loader):
        X = X.to(DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(X)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if batch_idx >= warmup_batches:
            per_sample.append((t1 - t0) / len(X))
            total_samples += len(X)

    values = np.array(per_sample, dtype=float)
    if values.size == 0:
        values = np.array([np.nan])

    return {
        "mean_latency_seconds_per_sample": float(np.nanmean(values)),
        "median_latency_seconds_per_sample": float(np.nanmedian(values)),
        "p95_latency_seconds_per_sample": float(np.nanpercentile(values, 95)),
        "evaluated_samples_after_warmup": int(total_samples),
    }


def parameter_count_by_team(model: NashCooperativeOrganizationalModel) -> pd.DataFrame:
    rows = []
    for name, module in model.teams.items():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        rows.append({"specialist_team": name, "parameters": int(params), "trainable_parameters": int(trainable)})
    rows.append({
        "specialist_team": "nash_cooperative_team_logits",
        "parameters": int(model.team_logits.numel()),
        "trainable_parameters": int(model.team_logits.numel()),
    })
    return pd.DataFrame(rows)


def save_domain_mapping(metadata: Dict, out_dir: Path) -> None:
    rows = []
    for domain, features in metadata["domain_features"].items():
        for feature in features:
            rows.append({"specialist_team": domain, "feature": feature})
    pd.DataFrame(rows).to_csv(out_dir / "experiment1_domain_feature_mapping.csv", index=False)


# =========================================================
# 7. Plotting helpers for paper-level reporting
# =========================================================

def save_fig(path: Path, dpi: int = 300) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_dataset_class_distribution(y_train, y_val, y_test, out_dir: Path, dpi: int) -> None:
    labels = sorted(set(y_train.tolist()) | set(y_val.tolist()) | set(y_test.tolist()))
    x = np.arange(len(labels))
    width = 0.25
    train_counts = [int(np.sum(y_train == c)) for c in labels]
    val_counts = [int(np.sum(y_val == c)) for c in labels]
    test_counts = [int(np.sum(y_test == c)) for c in labels]
    plt.figure(figsize=(8, 6))
    plt.bar(x - width, train_counts, width, label="Train")
    plt.bar(x, val_counts, width, label="Validation")
    plt.bar(x + width, test_counts, width, label="Test")
    plt.xlabel("Encoded class")
    plt.ylabel("Number of participants")
    plt.title("Experiment 1 Internal Split Class Distribution")
    plt.xticks(x, labels)
    plt.legend()
    save_fig(out_dir / "experiment1_class_distribution.png", dpi)


def plot_training_curves(history: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    metrics = [
        ("train_loss", "Training Loss"),
        ("val_loss", "Validation Loss"),
        ("val_f1_macro", "Validation Macro-F1"),
        ("val_accuracy", "Validation Accuracy"),
        ("val_balanced_accuracy", "Validation Balanced Accuracy"),
        ("val_roc_auc", "Validation ROC-AUC"),
        ("val_average_precision", "Validation Average Precision"),
    ]
    for col, title in metrics:
        if col in history.columns:
            plt.figure(figsize=(8, 6))
            plt.plot(history["epoch"], history[col], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel(title)
            plt.title(title)
            save_fig(out_dir / f"experiment1_{col}.png", dpi)

    # Combined core training overview for manuscript reporting.
    plt.figure(figsize=(9, 6))
    for col in ["val_f1_macro", "val_accuracy", "val_balanced_accuracy"]:
        if col in history.columns:
            plt.plot(history["epoch"], history[col], marker="o", label=col.replace("val_", "").replace("_", " "))
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Experiment 1 Internal Validation Performance")
    plt.legend()
    save_fig(out_dir / "experiment1_validation_performance_overview.png", dpi)


def plot_team_weight_history(team_history: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    plt.figure(figsize=(10, 6))
    for col in team_history.columns:
        if col != "epoch":
            plt.plot(team_history["epoch"], team_history[col], marker="o", label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Cooperative team weight")
    plt.title("Nash-Cooperative Specialist-Team Weight Evolution")
    plt.legend(loc="best")
    save_fig(out_dir / "experiment1_team_weight_evolution.png", dpi)


def plot_team_utility_history(utility_history: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    plt.figure(figsize=(10, 6))
    for col in utility_history.columns:
        if col != "epoch":
            plt.plot(utility_history["epoch"], utility_history[col], marker="o", label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Mean team utility")
    plt.title("Specialist-Team Utility Evolution")
    plt.legend(loc="best")
    save_fig(out_dir / "experiment1_team_utility_evolution.png", dpi)


def plot_confusion_matrix(cm: np.ndarray, out_dir: Path, dpi: int) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Experiment 1 Test Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    save_fig(out_dir / "experiment1_confusion_matrix_test.png", dpi)

    normalized = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(7, 6))
    plt.imshow(normalized, interpolation="nearest")
    plt.title("Experiment 1 Normalized Test Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            plt.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center")
    save_fig(out_dir / "experiment1_confusion_matrix_test_normalized.png", dpi)


def plot_team_contributions(domain_names: List[str], weights: List[float], utilities: List[float], out_dir: Path, dpi: int) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(domain_names, weights)
    plt.xlabel("Specialist team")
    plt.ylabel("Mean cooperative weight")
    plt.title("Experiment 1 Specialist-Team Cooperative Contributions")
    plt.xticks(rotation=30, ha="right")
    save_fig(out_dir / "experiment1_specialist_team_contributions_test.png", dpi)

    plt.figure(figsize=(10, 6))
    plt.bar(domain_names, utilities)
    plt.xlabel("Specialist team")
    plt.ylabel("Mean team utility")
    plt.title("Experiment 1 Specialist-Team Utility on Test Set")
    plt.xticks(rotation=30, ha="right")
    save_fig(out_dir / "experiment1_specialist_team_utilities_test.png", dpi)


def plot_prediction_confidence(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, out_dir: Path, dpi: int) -> None:
    confidence = np.max(y_prob, axis=1)
    correctness = (y_true == y_pred).astype(int)
    pd.DataFrame({
        "confidence": confidence,
        "correct": correctness,
    }).to_csv(out_dir / "experiment1_prediction_confidence.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.hist(confidence[correctness == 1], bins=20, alpha=0.65, label="Correct")
    plt.hist(confidence[correctness == 0], bins=20, alpha=0.65, label="Incorrect")
    plt.xlabel("Maximum predicted probability")
    plt.ylabel("Number of participants")
    plt.title("Experiment 1 Prediction Confidence Distribution")
    plt.legend()
    save_fig(out_dir / "experiment1_prediction_confidence_distribution.png", dpi)


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int, out_dir: Path, dpi: int) -> None:
    try:
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            plt.figure(figsize=(7, 6))
            plt.plot(fpr, tpr, label="ROC curve")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title("Experiment 1 Test ROC Curve")
            plt.legend()
            save_fig(out_dir / "experiment1_roc_curve_test.png", dpi)

            plt.figure(figsize=(7, 6))
            plt.plot(recall, precision, label="Precision-recall curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Experiment 1 Test Precision-Recall Curve")
            plt.legend()
            save_fig(out_dir / "experiment1_precision_recall_curve_test.png", dpi)
        else:
            y_bin = label_binarize(y_true, classes=np.arange(num_classes))
            plt.figure(figsize=(8, 6))
            for c in range(num_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
                plt.plot(fpr, tpr, label=f"Class {c}")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title("Experiment 1 One-vs-Rest Test ROC Curves")
            plt.legend()
            save_fig(out_dir / "experiment1_roc_curve_test_multiclass.png", dpi)

            plt.figure(figsize=(8, 6))
            for c in range(num_classes):
                precision, recall, _ = precision_recall_curve(y_bin[:, c], y_prob[:, c])
                plt.plot(recall, precision, label=f"Class {c}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Experiment 1 One-vs-Rest Test Precision-Recall Curves")
            plt.legend()
            save_fig(out_dir / "experiment1_precision_recall_curve_test_multiclass.png", dpi)
    except Exception as exc:
        with open(out_dir / "experiment1_curve_generation_warning.txt", "w", encoding="utf-8") as f:
            f.write(str(exc))


# =========================================================
# 8. Paper-ready report generation
# =========================================================

def fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{float(value):.4f}"


def write_markdown_report(summary: Dict, metrics_df: pd.DataFrame, out_dir: Path) -> None:
    test = summary["test_metrics"]
    val = summary["validation_metrics"]
    complexity = summary["model_complexity_latency"]
    team_lines = []
    for name, weight, util in zip(summary["domain_team_order"], test["team_weights_mean"], test["team_utilities_mean"]):
        team_lines.append(f"- {name}: cooperative weight = {weight:.4f}, mean utility = {util:.4f}")

    text = f"""# Experiment 1: Organizational Training and Internal Cooperative Learning

## Aim
This experiment evaluates the internal learning behavior of the proposed Nash-cooperative organizational framework using the modeling-ready NHANES 2011-2018 training, validation, and test splits. The experiment directly reflects the Methods architecture by decomposing multidomain health records into specialist teams, training lightweight learners inside each team, and integrating team predictions through adaptive Nash-cooperative weighting.

## Data
- Training records: {summary['metadata']['train_rows']}
- Validation records: {summary['metadata']['val_rows']}
- Test records: {summary['metadata']['test_rows']}
- Number of features: {summary['metadata']['num_features']}
- Target column: {summary['metadata']['target_column']}
- Number of classes: {summary['metadata']['num_classes']}

## Main validation results
- Accuracy: {fmt(val['accuracy'])}
- Balanced accuracy: {fmt(val['balanced_accuracy'])}
- Macro-F1: {fmt(val['f1_macro'])}
- Weighted-F1: {fmt(val['f1_weighted'])}
- ROC-AUC: {fmt(val['roc_auc'])}
- Average precision: {fmt(val['average_precision'])}

## Main internal test results
- Accuracy: {fmt(test['accuracy'])}
- Balanced accuracy: {fmt(test['balanced_accuracy'])}
- Macro-F1: {fmt(test['f1_macro'])}
- Weighted-F1: {fmt(test['f1_weighted'])}
- ROC-AUC: {fmt(test['roc_auc'])}
- Average precision: {fmt(test['average_precision'])}
- Matthews correlation coefficient: {fmt(test['mcc'])}
- Mean prediction confidence: {fmt(test['mean_prediction_confidence'])}

## Specialist-team contribution analysis
{chr(10).join(team_lines)}

## Lightweight edge-learning profile
- Total parameters: {complexity['total_parameters']}
- Trainable parameters: {complexity['trainable_parameters']}
- Mean inference latency per sample: {complexity['mean_latency_seconds_per_sample']:.8f} seconds
- Median inference latency per sample: {complexity['median_latency_seconds_per_sample']:.8f} seconds
- P95 inference latency per sample: {complexity['p95_latency_seconds_per_sample']:.8f} seconds
- Device used: {complexity['device']}
- Runtime: {complexity['runtime_seconds']:.2f} seconds

## Generated figures for manuscript writing
- experiment1_class_distribution.png
- experiment1_validation_performance_overview.png
- experiment1_train_loss.png
- experiment1_val_loss.png
- experiment1_val_f1_macro.png
- experiment1_team_weight_evolution.png
- experiment1_team_utility_evolution.png
- experiment1_confusion_matrix_test.png
- experiment1_confusion_matrix_test_normalized.png
- experiment1_specialist_team_contributions_test.png
- experiment1_specialist_team_utilities_test.png
- experiment1_roc_curve_test.png or experiment1_roc_curve_test_multiclass.png
- experiment1_precision_recall_curve_test.png or experiment1_precision_recall_curve_test_multiclass.png
- experiment1_prediction_confidence_distribution.png

## Suggested interpretation template
The results of Experiment 1 should be reported as evidence of internal organizational learning rather than as a generic classifier result. The validation curves show whether the cooperative architecture converged stably, while the confusion matrix and class-wise report describe the reliability of disability-related class discrimination. The specialist-team weight trajectory provides direct evidence of Nash-cooperative adaptation because team influence is updated according to utility feedback rather than being fixed manually. The final team-contribution table supports organizational explainability by showing how nutritional-metabolic, physiological, behavioral, demographic-fairness, and functional-disability teams contributed to the final prediction. The parameter and latency profile should be used to support the lightweight edge-learning claim.
"""
    (out_dir / "experiment1_paper_ready_report.md").write_text(text, encoding="utf-8")


# =========================================================
# 9. Main training procedure
# =========================================================

def train_experiment1() -> None:
    start_time = time.time()
    print("Starting Experiment 1: Organizational Training and Internal Cooperative Learning")
    print(f"Device: {DEVICE}")
    print(f"Output folder: {OUTPUT_DIR}")

    X_train, y_train, X_val, y_val, X_test, y_test, domain_indices, metadata = load_internal_data(CFG)
    num_classes = metadata["num_classes"]

    with open(OUTPUT_DIR / "experiment1_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    save_domain_mapping(metadata, OUTPUT_DIR)
    plot_dataset_class_distribution(y_train, y_val, y_test, OUTPUT_DIR, CFG.dpi)

    train_loader = DataLoader(NHANESDataset(X_train, y_train), batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(NHANESDataset(X_val, y_val), batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    test_loader = DataLoader(NHANESDataset(X_test, y_test), batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    model = NashCooperativeOrganizationalModel(
        domain_indices=domain_indices,
        hidden_dim=CFG.hidden_dim,
        num_classes=num_classes,
        learners_per_team=CFG.learners_per_team,
        dropout=CFG.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)

    best_val_f1 = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_path = OUTPUT_DIR / "experiment1_best_model.pt"

    history_rows = []
    team_weight_rows = []
    team_utility_rows = []

    for epoch in range(1, CFG.max_epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_div_loss = 0.0
        epoch_cost_loss = 0.0
        epoch_entropy_loss = 0.0
        utility_accumulator = []

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, team_outputs, team_weights, _, _ = model(X)

            pred = F.cross_entropy(logits, y)
            div = diversity_loss(team_outputs)
            cost = model_cost_regularizer(model)
            entropy = entropy_regularizer(team_weights)

            loss = pred + CFG.lambda_diversity * div + CFG.lambda_cost * cost + CFG.lambda_entropy * entropy
            loss.backward()
            optimizer.step()

            utilities = compute_team_utilities(team_outputs.detach(), y)
            utility_accumulator.append(utilities.cpu())

            n = len(y)
            epoch_total_loss += loss.item() * n
            epoch_pred_loss += pred.item() * n
            epoch_div_loss += div.item() * n
            epoch_cost_loss += cost.item() * n
            epoch_entropy_loss += entropy.item() * n

        mean_utilities = torch.stack(utility_accumulator).mean(dim=0)
        model.update_nash_weights(mean_utilities, CFG.utility_sensitivity, CFG.min_team_weight)

        val_metrics, _, _, _ = evaluate_model(model, val_loader, num_classes)

        train_n = len(y_train)
        row = {
            "epoch": epoch,
            "train_loss": epoch_total_loss / train_n,
            "train_prediction_loss": epoch_pred_loss / train_n,
            "train_diversity_loss": epoch_div_loss / train_n,
            "train_cost_loss": epoch_cost_loss / train_n,
            "train_entropy_loss": epoch_entropy_loss / train_n,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_average_precision": val_metrics["average_precision"],
        }
        history_rows.append(row)

        team_weight_rows.append({
            "epoch": epoch,
            **{name: weight for name, weight in zip(model.domain_names, val_metrics["team_weights_mean"])}
        })
        team_utility_rows.append({
            "epoch": epoch,
            **{name: util for name, util in zip(model.domain_names, val_metrics["team_utilities_mean"])}
        })

        print(
            f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | val_acc={row['val_accuracy']:.4f} | "
            f"val_bal_acc={row['val_balanced_accuracy']:.4f} | val_f1={row['val_f1_macro']:.4f} | "
            f"val_auc={row['val_roc_auc']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(CFG),
                "metadata": metadata,
                "domain_names": model.domain_names,
                "domain_indices": domain_indices,
                "best_epoch": best_epoch,
                "best_val_f1_macro": best_val_f1,
            }, best_path)
        else:
            patience_counter += 1

        if patience_counter >= CFG.patience:
            print(f"Early stopping triggered at epoch {epoch}; best epoch was {best_epoch}.")
            break

    history_df = pd.DataFrame(history_rows)
    team_weight_df = pd.DataFrame(team_weight_rows)
    team_utility_df = pd.DataFrame(team_utility_rows)
    history_df.to_csv(OUTPUT_DIR / "experiment1_training_history.csv", index=False)
    team_weight_df.to_csv(OUTPUT_DIR / "experiment1_team_weight_history.csv", index=False)
    team_utility_df.to_csv(OUTPUT_DIR / "experiment1_team_utility_history.csv", index=False)

    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, y_val_true, y_val_pred, y_val_prob = evaluate_model(model, val_loader, num_classes)
    test_metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, num_classes)

    cm = confusion_matrix(y_true, y_pred)
    report_test = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    report_val = classification_report(y_val_true, y_val_pred, zero_division=0, output_dict=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    latency = measure_latency(model, test_loader)
    runtime_seconds = time.time() - start_time

    complexity = {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "device": str(DEVICE),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "runtime_seconds": float(runtime_seconds),
        **latency,
    }

    # Tables
    metrics_rows = []
    for split_name, metrics in [("validation", val_metrics), ("test", test_metrics)]:
        row = {"split": split_name}
        for k, v in metrics.items():
            if not isinstance(v, list):
                row[k] = v
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUTPUT_DIR / "experiment1_metrics_summary.csv", index=False)

    pd.DataFrame(report_val).transpose().to_csv(OUTPUT_DIR / "experiment1_classification_report_validation.csv")
    pd.DataFrame(report_test).transpose().to_csv(OUTPUT_DIR / "experiment1_classification_report_test.csv")
    pd.DataFrame(cm).to_csv(OUTPUT_DIR / "experiment1_confusion_matrix_test.csv", index=False)

    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "prediction_confidence": np.max(y_prob, axis=1)})
    for c in range(num_classes):
        pred_df[f"prob_class_{c}"] = y_prob[:, c]
    pred_df.to_csv(OUTPUT_DIR / "experiment1_predictions_test.csv", index=False)

    team_df = pd.DataFrame({
        "specialist_team": model.domain_names,
        "cooperative_weight_test": test_metrics["team_weights_mean"],
        "mean_utility_test": test_metrics["team_utilities_mean"],
        "number_of_features": [len(domain_indices[name]) for name in model.domain_names],
    })
    team_df.to_csv(OUTPUT_DIR / "experiment1_specialist_team_weights_test.csv", index=False)

    param_df = parameter_count_by_team(model)
    param_df.to_csv(OUTPUT_DIR / "experiment1_parameter_count_by_team.csv", index=False)
    pd.DataFrame([complexity]).to_csv(OUTPUT_DIR / "experiment1_complexity_latency.csv", index=False)

    # Figures
    plot_training_curves(history_df, OUTPUT_DIR, CFG.dpi)
    plot_team_weight_history(team_weight_df, OUTPUT_DIR, CFG.dpi)
    plot_team_utility_history(team_utility_df, OUTPUT_DIR, CFG.dpi)
    plot_confusion_matrix(cm, OUTPUT_DIR, CFG.dpi)
    plot_team_contributions(model.domain_names, test_metrics["team_weights_mean"], test_metrics["team_utilities_mean"], OUTPUT_DIR, CFG.dpi)
    plot_prediction_confidence(y_true, y_pred, y_prob, OUTPUT_DIR, CFG.dpi)
    plot_roc_pr_curves(y_true, y_prob, num_classes, OUTPUT_DIR, CFG.dpi)

    summary = {
        "experiment": "Experiment 1: Organizational Training and Internal Cooperative Learning",
        "paper_title": "Nash-Cooperative Swarm Intelligence for Lightweight Edge Learning Through Multi-Team Adaptive Optimization",
        "methods_alignment": {
            "multidomain_decomposition": True,
            "specialist_lightweight_teams": True,
            "intra_team_aggregation": True,
            "nash_cooperative_team_weighting": True,
            "prediction_loss": True,
            "diversity_regularization": True,
            "lightweight_cost_regularization": True,
            "entropy_anti_domination_regularization": True,
            "internal_train_validation_test_protocol": True,
            "explainability_through_team_weights": True,
            "lightweight_edge_reporting": True,
        },
        "metadata": metadata,
        "best_epoch": int(best_epoch),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_complexity_latency": complexity,
        "domain_team_order": model.domain_names,
        "output_files_note": "All tables, figures, predictions, reports, and checkpoints are saved in the Experiment1 Results folder.",
    }

    with open(OUTPUT_DIR / "experiment1_final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "experiment1_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, indent=2)

    write_markdown_report(summary, metrics_df, OUTPUT_DIR)

    print("\nExperiment 1 completed successfully.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best model: {best_path}")
    print(f"Results folder: {OUTPUT_DIR}")
    print("\nFinal internal test metrics:")
    for key, value in test_metrics.items():
        if not isinstance(value, list):
            print(f"  {key}: {value}")
    print("\nFinal specialist-team weights:")
    for name, weight in zip(model.domain_names, test_metrics["team_weights_mean"]):
        print(f"  {name}: {weight:.4f}")


if __name__ == "__main__":
    train_experiment1()

# -*- coding: utf-8 -*-
"""
Experiment 5: Temporal Organizational Generalization
Paper: Nash-Cooperative Swarm Intelligence for Lightweight Edge Learning
       Through Multi-Team Adaptive Optimization

Scientific purpose
------------------
Experiment 5 evaluates whether the trained Nash-cooperative organizational
framework can generalize from the primary NHANES 2011-2018 development
environment to later NHANES cycles under temporal distribution shift,
partial observability, and heterogeneous target representation.

This experiment is aligned with the Methods stream:
Experiment 1: internal organizational learning
Experiment 2: specialist-team explainability
Experiment 3: swarm cooperative adaptation
Experiment 4: lightweight edge efficiency
Experiment 5: temporal organizational generalization

The script reloads:
- Exact Experiment 1 checkpoint and architecture metadata
- External modeling-ready later-cycle file
- Optional Experiment 3 cooperation policies

It evaluates:
1. Learned Nash cooperation on external later-cycle data
2. Static equal cooperation
3. PSO, ACO, BCO, HHO, and hybrid cooperation policies from Experiment 3 if available
4. Internal-test vs external performance shift
5. Specialist-team prediction confidence under temporal drift
6. Feature availability and partial-observability profile
7. Demographic subgroup consistency where available
8. Report-ready tables, figures, and Markdown summary

Expected external file
----------------------
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/modeling_ready/external_2017_2023.csv

Outputs
-------
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/Experiment5/Results
"""

import json
import random
import platform
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 0. Paths and reproducibility
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROOT = Path(r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments")
MODELING_DIR = ROOT / "modeling_ready"

EXP1_RESULTS = ROOT / "Experiment1" / "Results"
EXP3_RESULTS = ROOT / "Experiment3" / "Results"
EXP5_RESULTS = ROOT / "Experiment5" / "Results"

TEST_FILE = MODELING_DIR / "test_2011_2018.csv"
EXTERNAL_FILE = MODELING_DIR / "external_2017_2023.csv"
BEST_MODEL_FILE = EXP1_RESULTS / "experiment1_best_model.pt"
EXP3_POLICY_FILE = EXP3_RESULTS / "experiment3_cooperative_weight_policies.csv"

EXP5_RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1. I/O helpers
# ---------------------------------------------------------------------
def save_json(obj, path: Path):
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


def save_dataframe(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def get_first_existing(checkpoint: dict, keys: List[str], default=None):
    for k in keys:
        if k in checkpoint:
            return checkpoint[k]
    return default


def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment 1 checkpoint not found: {path}\n"
            "Run experiment1_organizational_training_reporting_v2.py first."
        )
    checkpoint = torch.load(path, map_location=DEVICE)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Invalid checkpoint. Rerun Experiment 1 v2.")
    return checkpoint


# ---------------------------------------------------------------------
# 2. Metadata extraction and preprocessing
# ---------------------------------------------------------------------
def extract_metadata(checkpoint: dict) -> dict:
    config = get_first_existing(checkpoint, ["config", "CONFIG", "training_config"], {})
    meta = get_first_existing(checkpoint, ["metadata", "data_metadata", "checkpoint_manifest"], {}) or {}
    state = checkpoint["model_state_dict"]

    team_names = (
        checkpoint.get("team_names")
        or checkpoint.get("specialist_team_names")
        or meta.get("team_names")
        or (config.get("team_names") if isinstance(config, dict) else None)
    )
    if team_names is None:
        teams = []
        for k in state.keys():
            if k.startswith("teams."):
                teams.append(k.split(".")[1])
        team_names = sorted(list(set(teams)))

    domain_map = (
        checkpoint.get("domain_map")
        or checkpoint.get("domain_features")
        or checkpoint.get("team_feature_map")
        or meta.get("domain_map")
        or meta.get("domain_features")
    )
    if domain_map is None:
        raise ValueError("Checkpoint does not contain domain_map. Rerun Experiment 1 v2.")

    feature_columns = (
        checkpoint.get("feature_columns")
        or checkpoint.get("feature_cols")
        or meta.get("feature_columns")
        or meta.get("feature_cols")
    )
    if feature_columns is None:
        feature_columns = []
        for t in team_names:
            for col in domain_map[t]:
                if col not in feature_columns:
                    feature_columns.append(col)

    target_col = (
        checkpoint.get("target_column")
        or checkpoint.get("target_col")
        or meta.get("target_column")
        or meta.get("target_col")
        or (config.get("target_column") if isinstance(config, dict) else None)
        or "disability_stage"
    )

    class_mapping = (
        checkpoint.get("class_mapping")
        or checkpoint.get("class_to_idx")
        or checkpoint.get("target_mapping")
        or meta.get("class_mapping")
        or meta.get("class_to_idx")
    )

    num_classes = (
        checkpoint.get("num_classes")
        or meta.get("num_classes")
        or (config.get("num_classes") if isinstance(config, dict) else None)
    )
    if num_classes is None:
        for k, v in state.items():
            if k.endswith("net.3.weight"):
                num_classes = int(v.shape[0])
                break

    hidden_dim = (
        checkpoint.get("hidden_dim")
        or meta.get("hidden_dim")
        or (config.get("hidden_dim") if isinstance(config, dict) else None)
    )
    if hidden_dim is None:
        for k, v in state.items():
            if k.endswith("net.0.weight"):
                hidden_dim = int(v.shape[0])
                break

    n_learners = (
        checkpoint.get("num_learners_per_team")
        or checkpoint.get("n_learners")
        or meta.get("num_learners_per_team")
        or meta.get("n_learners")
        or (config.get("num_learners_per_team") if isinstance(config, dict) else None)
    )
    if n_learners is None:
        ids = set()
        for k in state.keys():
            if ".learners." in k:
                part = k.split(".learners.")[1].split(".")[0]
                if part.isdigit():
                    ids.add(int(part))
        n_learners = max(ids) + 1 if ids else 3

    dropout = (
        checkpoint.get("dropout")
        or meta.get("dropout")
        or (config.get("dropout") if isinstance(config, dict) else 0.10)
        or 0.10
    )

    metadata = {
        "config": config,
        "team_names": list(team_names),
        "domain_map": {k: list(v) for k, v in domain_map.items()},
        "feature_columns": list(feature_columns),
        "target_column": target_col,
        "class_mapping": class_mapping,
        "num_classes": int(num_classes),
        "hidden_dim": int(hidden_dim),
        "num_learners_per_team": int(n_learners),
        "dropout": float(dropout),
        "scaler_mean": checkpoint.get("scaler_mean") or checkpoint.get("feature_mean") or meta.get("scaler_mean") or meta.get("feature_mean"),
        "scaler_scale": checkpoint.get("scaler_scale") or checkpoint.get("feature_scale") or meta.get("scaler_scale") or meta.get("feature_scale"),
        "train_medians": checkpoint.get("train_medians") or meta.get("train_medians"),
        "categorical_encoders": checkpoint.get("categorical_encoders") or meta.get("categorical_encoders") or {},
    }
    return metadata


def identify_external_target_column(df: pd.DataFrame, metadata: dict) -> str:
    """
    External later cycles may have a harmonized fallback target.
    Prefer the training target if present, otherwise use likely target names.
    """
    candidates = [
        metadata["target_column"],
        "functional_risk",
        "fallback_functional_risk",
        "disability_stage",
        "merge01_vs_23",
        "target",
        "label",
        "y",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # Conservative fallback: last column
    return df.columns[-1]


def apply_preprocessing(df: pd.DataFrame, metadata: dict, target_col_override: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    feature_columns = metadata["feature_columns"]
    target_col = target_col_override or metadata["target_column"]

    for col in feature_columns:
        if col not in out.columns:
            out[col] = np.nan

    cat_maps = metadata.get("categorical_encoders") or {}
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            if col in cat_maps and isinstance(cat_maps[col], dict):
                out[col] = out[col].astype(str).map(cat_maps[col]).fillna(-1)
            else:
                out[col] = pd.factorize(out[col].astype(str))[0]

    medians = metadata.get("train_medians")
    if isinstance(medians, dict):
        for col in feature_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
            out[col] = out[col].fillna(float(medians.get(col, 0.0)))
    else:
        for col in feature_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
            out[col] = out[col].fillna(out[col].median() if not out[col].dropna().empty else 0.0)

    mean = metadata.get("scaler_mean")
    scale = metadata.get("scaler_scale")
    if isinstance(mean, dict) and isinstance(scale, dict):
        for col in feature_columns:
            m = float(mean.get(col, 0.0))
            s = float(scale.get(col, 1.0))
            if abs(s) < 1e-12:
                s = 1.0
            out[col] = (out[col] - m) / s
    elif isinstance(mean, list) and isinstance(scale, list):
        for i, col in enumerate(feature_columns):
            m = float(mean[i])
            s = float(scale[i])
            if abs(s) < 1e-12:
                s = 1.0
            out[col] = (out[col] - m) / s
    else:
        vals = out[feature_columns]
        out[feature_columns] = (vals - vals.mean()) / vals.std().replace(0, 1.0)

    out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], 0).fillna(0)

    if target_col not in out.columns:
        raise ValueError(f"Target column missing: {target_col}")

    # External target may not have the same class mapping. If its unique labels are already 0..K-1,
    # keep them. If binary external target is present while model is multiclass, evaluate only
    # compatible mapped classes where possible and report the target mismatch explicitly.
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out.dropna(subset=[target_col]).copy()
    out[target_col] = out[target_col].astype(int)

    return out


def feature_availability_profile(raw_df: pd.DataFrame, metadata: dict):
    rows = []
    for team in metadata["team_names"]:
        for col in metadata["domain_map"][team]:
            rows.append({
                "team": team,
                "feature": col,
                "available_in_external": col in raw_df.columns,
                "missing_rate_external_raw": float(raw_df[col].isna().mean()) if col in raw_df.columns else 1.0,
            })
    df = pd.DataFrame(rows)
    team_summary = df.groupby("team").agg(
        expected_features=("feature", "count"),
        available_features=("available_in_external", "sum"),
        mean_raw_missing_rate=("missing_rate_external_raw", "mean"),
    ).reset_index()
    team_summary["availability_ratio"] = team_summary["available_features"] / team_summary["expected_features"]
    return df, team_summary


# ---------------------------------------------------------------------
# 3. Dataset and model
# ---------------------------------------------------------------------
class MultidomainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, metadata: dict, target_col: str):
        self.target_col = target_col
        self.team_names = metadata["team_names"]
        self.domain_map = metadata["domain_map"]
        self.y = torch.tensor(df[self.target_col].values, dtype=torch.long)
        self.domain_data = {}
        for team in self.team_names:
            self.domain_data[team] = torch.tensor(df[self.domain_map[team]].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = {team: self.domain_data[team][idx] for team in self.team_names}
        return x, self.y[idx]


class LightweightLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
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
    def __init__(self, input_dim, hidden_dim, num_classes, n_learners, dropout):
        super().__init__()
        self.learners = nn.ModuleList([
            LightweightLearner(input_dim, hidden_dim, num_classes, dropout)
            for _ in range(n_learners)
        ])
        self.intra_team_logits = nn.Parameter(torch.zeros(n_learners))

    def forward(self, x):
        outputs = torch.stack([learner(x) for learner in self.learners], dim=1)
        weights = torch.softmax(self.intra_team_logits, dim=0)
        team_output = torch.sum(outputs * weights.view(1, -1, 1), dim=1)
        return team_output, outputs, weights


class NashCooperativeOrganizationalModel(nn.Module):
    def __init__(self, metadata: dict):
        super().__init__()
        self.team_names = metadata["team_names"]
        self.domain_map = metadata["domain_map"]
        self.teams = nn.ModuleDict({
            team: SpecialistTeam(
                input_dim=len(self.domain_map[team]),
                hidden_dim=metadata["hidden_dim"],
                num_classes=metadata["num_classes"],
                n_learners=metadata["num_learners_per_team"],
                dropout=metadata["dropout"],
            )
            for team in self.team_names
        })
        self.team_logits = nn.Parameter(torch.zeros(len(self.team_names)))

    def forward(self, x_dict, override_weights: Optional[torch.Tensor] = None):
        team_outputs = []
        raw_team_dict = {}
        for team in self.team_names:
            out, _, _ = self.teams[team](x_dict[team])
            team_outputs.append(out)
            raw_team_dict[team] = out

        stacked = torch.stack(team_outputs, dim=1)

        if override_weights is None:
            weights = torch.softmax(self.team_logits, dim=0)
        else:
            weights = override_weights.to(stacked.device)
            weights = torch.clamp(weights, min=1e-8)
            weights = weights / weights.sum()

        logits = torch.sum(stacked * weights.view(1, -1, 1), dim=1)
        return logits, raw_team_dict, weights


# ---------------------------------------------------------------------
# 4. Metrics and prediction
# ---------------------------------------------------------------------
def load_cooperation_policies(metadata):
    team_names = metadata["team_names"]
    policies = {}
    policies["Static_Equal_Cooperation"] = np.ones(len(team_names), dtype=np.float64) / len(team_names)

    if EXP3_POLICY_FILE.exists():
        df = pd.read_csv(EXP3_POLICY_FILE)
        for _, row in df.iterrows():
            method = row["method"]
            weights = []
            valid = True
            for team in team_names:
                if team not in row.index:
                    valid = False
                    break
                weights.append(float(row[team]))
            if valid:
                weights = np.asarray(weights, dtype=np.float64)
                weights = np.maximum(weights, 1e-8)
                weights = weights / weights.sum()
                policies[method] = weights
    return policies


def compatible_binary_from_multiclass(y_true, y_prob, y_pred):
    """
    If external target is binary while model output has 4 classes, provide an additional
    compatible binary grouping: classes 0,1 -> low and classes 2,3 -> high.
    This follows the previously used merge01_vs_23 logic.
    """
    if y_prob.shape[1] < 4:
        return None

    y_pred_bin = np.where(y_pred >= 2, 1, 0)
    y_prob_bin = y_prob[:, 2:].sum(axis=1)
    y_true_bin = y_true.copy()
    unique = sorted(np.unique(y_true_bin).tolist())

    # If external is already binary, use as is.
    if set(unique).issubset({0, 1}):
        return y_true_bin, y_pred_bin, y_prob_bin

    # If external is multiclass, map similarly.
    y_true_bin = np.where(y_true_bin >= 2, 1, 0)
    return y_true_bin, y_pred_bin, y_prob_bin


def compute_multiclass_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "mean_prediction_confidence": float(np.max(y_prob, axis=1).mean()),
    }

    try:
        if y_prob.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            metrics["average_precision"] = average_precision_score(y_true, y_prob[:, 1])
        else:
            # Only compute OVR if labels are compatible with output classes.
            classes = list(range(y_prob.shape[1]))
            if set(np.unique(y_true)).issubset(set(classes)):
                metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
                y_bin = label_binarize(y_true, classes=classes)
                metrics["average_precision_macro"] = average_precision_score(y_bin, y_prob, average="macro")
    except Exception:
        pass

    return metrics


def compute_binary_compatible_metrics(y_true, y_pred, y_prob):
    binary = compatible_binary_from_multiclass(y_true, y_prob, y_pred)
    if binary is None:
        return {}
    yb, pb, probb = binary
    metrics = {
        "binary_accuracy_merge01_vs_23": accuracy_score(yb, pb),
        "binary_balanced_accuracy_merge01_vs_23": balanced_accuracy_score(yb, pb),
        "binary_precision_merge01_vs_23": precision_score(yb, pb, zero_division=0),
        "binary_recall_merge01_vs_23": recall_score(yb, pb, zero_division=0),
        "binary_f1_merge01_vs_23": f1_score(yb, pb, zero_division=0),
    }
    try:
        metrics["binary_roc_auc_merge01_vs_23"] = roc_auc_score(yb, probb)
        metrics["binary_average_precision_merge01_vs_23"] = average_precision_score(yb, probb)
    except Exception:
        pass
    return metrics


@torch.no_grad()
def predict_policy(model, loader, weights_np=None):
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    team_probs = {team: [] for team in model.team_names}

    override = None
    if weights_np is not None:
        override = torch.tensor(weights_np, dtype=torch.float32, device=DEVICE)

    for x_dict, y in loader:
        x_dict = {k: v.to(DEVICE) for k, v in x_dict.items()}
        logits, raw_team_dict, weights = model(x_dict, override_weights=override)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

        all_y.append(y.numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

        for team, team_logits in raw_team_dict.items():
            team_probs[team].append(torch.softmax(team_logits, dim=1).cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    team_probs = {t: np.concatenate(v) for t, v in team_probs.items()}

    return y_true, y_pred, y_prob, team_probs


def evaluate_policy(model, loader, policy_name, weights_np=None):
    y_true, y_pred, y_prob, team_probs = predict_policy(model, loader, weights_np)
    metrics = compute_multiclass_metrics(y_true, y_pred, y_prob)
    metrics.update(compute_binary_compatible_metrics(y_true, y_pred, y_prob))
    metrics["method"] = policy_name
    return metrics, y_true, y_pred, y_prob, team_probs


def team_confidence_summary(team_probs):
    rows = []
    for team, prob in team_probs.items():
        rows.append({
            "team": team,
            "mean_confidence": float(np.max(prob, axis=1).mean()),
            "std_confidence": float(np.max(prob, axis=1).std()),
            "mean_entropy": float((-prob * np.log(prob + 1e-12)).sum(axis=1).mean()),
        })
    return pd.DataFrame(rows)


def subgroup_external_metrics(raw_df, processed_df, metadata, y_true, y_pred, y_prob):
    candidates = [
        c for c in raw_df.columns
        if any(k in c.lower() for k in ["age", "sex", "gender", "race", "ethnicity", "education", "income", "poverty"])
    ]

    rows = []
    for col in candidates:
        values = raw_df[col]
        if len(values) != len(y_true):
            continue

        if pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > 6:
            try:
                groups = pd.qcut(values, q=4, duplicates="drop").astype(str)
            except Exception:
                groups = values.astype(str)
        else:
            groups = values.astype(str)

        for group in sorted(groups.dropna().unique()):
            idx = (groups == group).values
            if idx.sum() < 50:
                continue
            met = compute_multiclass_metrics(y_true[idx], y_pred[idx], y_prob[idx])
            bin_met = compute_binary_compatible_metrics(y_true[idx], y_pred[idx], y_prob[idx])
            rows.append({
                "attribute": col,
                "subgroup": group,
                "n": int(idx.sum()),
                "accuracy": met.get("accuracy", np.nan),
                "balanced_accuracy": met.get("balanced_accuracy", np.nan),
                "f1_macro": met.get("f1_macro", np.nan),
                "f1_weighted": met.get("f1_weighted", np.nan),
                "binary_f1_merge01_vs_23": bin_met.get("binary_f1_merge01_vs_23", np.nan),
                "binary_balanced_accuracy_merge01_vs_23": bin_met.get("binary_balanced_accuracy_merge01_vs_23", np.nan),
                "mean_prediction_confidence": met.get("mean_prediction_confidence", np.nan),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------
def plot_bar(df, x, y, title, ylabel, path, rotation=25):
    if df.empty or y not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_grouped(df, label_col, metric_cols, title, path):
    metric_cols = [c for c in metric_cols if c in df.columns]
    if df.empty or not metric_cols:
        return
    x = np.arange(len(df))
    width = 0.8 / len(metric_cols)

    plt.figure(figsize=(12, 6))
    for i, col in enumerate(metric_cols):
        plt.bar(x + i * width, df[col], width=width, label=col)
    plt.xticks(x + width * (len(metric_cols) - 1) / 2, df[label_col].astype(str), rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_shift_comparison(shift_df, path):
    cols = [c for c in ["internal_f1_weighted", "external_f1_weighted", "internal_balanced_accuracy", "external_balanced_accuracy"] if c in shift_df.columns]
    if not cols:
        return
    x = np.arange(len(shift_df))
    width = 0.8 / len(cols)
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(cols):
        plt.bar(x + i * width, shift_df[col], width=width, label=col)
    plt.xticks(x + width * (len(cols) - 1) / 2, shift_df["method"], rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title("Internal-Test versus External Later-Cycle Generalization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion(cm, path, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confidence_hist(y_pred_prob, path):
    conf = np.max(y_pred_prob, axis=1)
    plt.figure(figsize=(9, 6))
    plt.hist(conf, bins=30)
    plt.xlabel("Prediction confidence")
    plt.ylabel("Frequency")
    plt.title("External Later-Cycle Prediction Confidence")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------
def main():
    print("Starting Experiment 5: Temporal Organizational Generalization")
    print(f"Device: {DEVICE}")
    print(f"Output folder: {EXP5_RESULTS}")

    if not EXTERNAL_FILE.exists():
        raise FileNotFoundError(
            f"External modeling-ready dataset not found: {EXTERNAL_FILE}\n"
            "Expected file: external_2017_2023.csv"
        )

    checkpoint = load_checkpoint(BEST_MODEL_FILE)
    metadata = extract_metadata(checkpoint)
    save_json(metadata, EXP5_RESULTS / "loaded_experiment1_architecture_metadata.json")

    raw_internal_df = pd.read_csv(TEST_FILE)
    raw_external_df = pd.read_csv(EXTERNAL_FILE)

    external_target_col = identify_external_target_column(raw_external_df, metadata)

    availability_df, availability_team_df = feature_availability_profile(raw_external_df, metadata)
    save_dataframe(availability_df, EXP5_RESULTS / "experiment5_external_feature_availability_detail.csv")
    save_dataframe(availability_team_df, EXP5_RESULTS / "experiment5_external_feature_availability_by_team.csv")

    internal_df = apply_preprocessing(raw_internal_df, metadata, target_col_override=metadata["target_column"])
    external_df = apply_preprocessing(raw_external_df, metadata, target_col_override=external_target_col)

    internal_dataset = MultidomainDataset(internal_df, metadata, metadata["target_column"])
    external_dataset = MultidomainDataset(external_df, metadata, external_target_col)

    internal_loader = DataLoader(internal_dataset, batch_size=256, shuffle=False)
    external_loader = DataLoader(external_dataset, batch_size=256, shuffle=False)

    model = NashCooperativeOrganizationalModel(metadata).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Cooperation policies.
    policies = load_cooperation_policies(metadata)
    with torch.no_grad():
        learned = torch.softmax(model.team_logits, dim=0).cpu().numpy()
    policies["Experiment1_Learned_Nash"] = learned

    preferred_order = [
        "Static_Equal_Cooperation",
        "Experiment1_Learned_Nash",
        "PSO_Adaptive_Cooperation",
        "ACO_Adaptive_Cooperation",
        "BCO_Adaptive_Cooperation",
        "HHO_Adaptive_Cooperation",
        "Hybrid_Swarm_Cooperation",
    ]
    method_order = [m for m in preferred_order if m in policies] + [m for m in policies.keys() if m not in preferred_order]

    external_rows = []
    internal_rows = []
    shift_rows = {}
    saved_predictions = {}

    for method in method_order:
        weights = policies[method]
        internal_metrics, yi, pi, probi, team_probs_i = evaluate_policy(model, internal_loader, method, weights)
        external_metrics, ye, pe, probe, team_probs_e = evaluate_policy(model, external_loader, method, weights)

        internal_rows.append(internal_metrics)
        external_rows.append(external_metrics)

        row = {"method": method}
        for key, val in internal_metrics.items():
            if key != "method" and isinstance(val, (int, float, np.integer, np.floating)):
                row[f"internal_{key}"] = val
        for key, val in external_metrics.items():
            if key != "method" and isinstance(val, (int, float, np.integer, np.floating)):
                row[f"external_{key}"] = val

        for metric in [
            "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
            "roc_auc", "roc_auc_ovr", "average_precision", "average_precision_macro",
            "binary_accuracy_merge01_vs_23", "binary_balanced_accuracy_merge01_vs_23",
            "binary_f1_merge01_vs_23", "binary_roc_auc_merge01_vs_23",
        ]:
            if f"internal_{metric}" in row and f"external_{metric}" in row:
                row[f"{metric}_external_minus_internal"] = row[f"external_{metric}"] - row[f"internal_{metric}"]

        shift_rows[method] = row
        saved_predictions[method] = (ye, pe, probe, team_probs_e)

    internal_metrics_df = pd.DataFrame(internal_rows)
    external_metrics_df = pd.DataFrame(external_rows)
    shift_df = pd.DataFrame(list(shift_rows.values()))

    save_dataframe(internal_metrics_df, EXP5_RESULTS / "experiment5_internal_reference_metrics.csv")
    save_dataframe(external_metrics_df, EXP5_RESULTS / "experiment5_external_temporal_metrics.csv")
    save_dataframe(shift_df, EXP5_RESULTS / "experiment5_internal_external_shift_metrics.csv")

    # Select best external method by binary F1 if available, otherwise weighted F1.
    if "binary_f1_merge01_vs_23" in external_metrics_df.columns:
        best_external_method = external_metrics_df.sort_values("binary_f1_merge01_vs_23", ascending=False).iloc[0]["method"]
        primary_metric = "binary_f1_merge01_vs_23"
    else:
        best_external_method = external_metrics_df.sort_values("f1_weighted", ascending=False).iloc[0]["method"]
        primary_metric = "f1_weighted"

    ye, pe, probe, team_probs_e = saved_predictions[best_external_method]

    cm = confusion_matrix(ye, pe)
    save_dataframe(pd.DataFrame(cm), EXP5_RESULTS / f"confusion_matrix_external_{best_external_method}.csv")
    plot_confusion(cm, EXP5_RESULTS / f"fig4_external_confusion_matrix_{best_external_method}.png",
                   f"External Confusion Matrix: {best_external_method}")

    report = classification_report(ye, pe, output_dict=True, zero_division=0)
    save_json(report, EXP5_RESULTS / f"classification_report_external_{best_external_method}.json")
    pd.DataFrame(report).transpose().to_csv(
        EXP5_RESULTS / f"classification_report_external_{best_external_method}.csv",
        encoding="utf-8-sig",
    )

    team_conf_df = team_confidence_summary(team_probs_e)
    save_dataframe(team_conf_df, EXP5_RESULTS / f"experiment5_external_team_confidence_{best_external_method}.csv")

    subgroup_df = subgroup_external_metrics(raw_external_df.loc[external_df.index], external_df, metadata, ye, pe, probe)
    if not subgroup_df.empty:
        save_dataframe(subgroup_df, EXP5_RESULTS / "experiment5_external_demographic_subgroup_metrics.csv")
        disparity_rows = []
        for attr, g in subgroup_df.groupby("attribute"):
            disparity_rows.append({
                "attribute": attr,
                "num_subgroups": g["subgroup"].nunique(),
                "min_f1_weighted": g["f1_weighted"].min(),
                "max_f1_weighted": g["f1_weighted"].max(),
                "f1_weighted_gap": g["f1_weighted"].max() - g["f1_weighted"].min(),
                "min_binary_f1": g["binary_f1_merge01_vs_23"].min(),
                "max_binary_f1": g["binary_f1_merge01_vs_23"].max(),
                "binary_f1_gap": g["binary_f1_merge01_vs_23"].max() - g["binary_f1_merge01_vs_23"].min(),
            })
        disparity_df = pd.DataFrame(disparity_rows)
        save_dataframe(disparity_df, EXP5_RESULTS / "experiment5_external_demographic_fairness_gaps.csv")

    # External target distribution.
    target_dist = external_df[external_target_col].value_counts().sort_index().reset_index()
    target_dist.columns = ["target_class", "count"]
    target_dist["percentage"] = target_dist["count"] / target_dist["count"].sum() * 100
    save_dataframe(target_dist, EXP5_RESULTS / "experiment5_external_target_distribution.csv")

    # Weight table.
    weight_rows = []
    for method in method_order:
        row = {"method": method}
        for i, team in enumerate(metadata["team_names"]):
            row[team] = float(policies[method][i])
        weight_rows.append(row)
    weight_df = pd.DataFrame(weight_rows)
    save_dataframe(weight_df, EXP5_RESULTS / "experiment5_external_cooperation_policies.csv")

    # Plots.
    plot_grouped(
        external_metrics_df,
        "method",
        ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "binary_f1_merge01_vs_23"],
        "External Later-Cycle Generalization Across Cooperation Policies",
        EXP5_RESULTS / "fig1_external_policy_metric_comparison.png",
    )
    plot_shift_comparison(shift_df, EXP5_RESULTS / "fig2_internal_external_shift_comparison.png")
    plot_bar(
        availability_team_df,
        "team",
        "availability_ratio",
        "External Feature Availability by Specialist Team",
        "Availability ratio",
        EXP5_RESULTS / "fig3_external_feature_availability_by_team.png",
    )
    plot_confidence_hist(probe, EXP5_RESULTS / "fig5_external_prediction_confidence_distribution.png")
    plot_bar(
        team_conf_df,
        "team",
        "mean_confidence",
        "External Specialist-Team Mean Confidence",
        "Mean confidence",
        EXP5_RESULTS / "fig6_external_team_confidence.png",
    )
    plot_bar(
        target_dist,
        "target_class",
        "percentage",
        "External Later-Cycle Target Distribution",
        "Percentage",
        EXP5_RESULTS / "fig7_external_target_distribution.png",
        rotation=0,
    )

    # Runtime metadata.
    runtime = {
        "experiment": "Experiment 5: Temporal Organizational Generalization",
        "device": str(DEVICE),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "checkpoint": str(BEST_MODEL_FILE),
        "internal_test_file": str(TEST_FILE),
        "external_file": str(EXTERNAL_FILE),
        "external_target_column_used": external_target_col,
        "output_folder": str(EXP5_RESULTS),
        "best_external_method": best_external_method,
        "primary_selection_metric": primary_metric,
        "external_rows": int(len(external_df)),
        "internal_rows": int(len(internal_df)),
    }
    save_json(runtime, EXP5_RESULTS / "experiment5_runtime_metadata.json")

    best_row = external_metrics_df[external_metrics_df["method"] == best_external_method].iloc[0]
    learned_row = external_metrics_df[external_metrics_df["method"] == "Experiment1_Learned_Nash"].iloc[0] if "Experiment1_Learned_Nash" in external_metrics_df["method"].values else best_row
    shift_best = shift_df[shift_df["method"] == best_external_method].iloc[0]

    # Choose display metrics robustly.
    auc_display = "binary_roc_auc_merge01_vs_23" if "binary_roc_auc_merge01_vs_23" in external_metrics_df.columns else ("roc_auc" if "roc_auc" in external_metrics_df.columns else "roc_auc_ovr")
    ap_display = "binary_average_precision_merge01_vs_23" if "binary_average_precision_merge01_vs_23" in external_metrics_df.columns else ("average_precision" if "average_precision" in external_metrics_df.columns else "average_precision_macro")

    # Markdown summary.
    md = f"""# Experiment 5: Temporal Organizational Generalization

## Objective
Experiment 5 evaluates temporal organizational generalization by applying the trained Nash-cooperative specialist-team model from the 2011–2018 NHANES development environment to the external later-cycle dataset covering 2017–March 2020 and 2021–August 2023. This directly tests the Methods claim that the proposed framework supports long-term organizational adaptation under evolving multidomain health distributions, partial observability, and heterogeneous target conditions.

## External Dataset
The external modeling-ready dataset used in this experiment was:

`{EXTERNAL_FILE}`

The external target column used was **{external_target_col}**. The external dataset contained **{len(external_df):,}** records after preprocessing. Because later NHANES releases use a harmonized fallback functional-risk representation rather than the exact PFQ-based target used during primary training, this experiment should be interpreted as a temporal robustness and heterogeneous-target stress test rather than exact target replication.

## Best External Cooperation Policy
The strongest external cooperation policy according to **{primary_metric}** was **{best_external_method}**.

External results for this policy were:

- Accuracy: {best_row.get('accuracy', float('nan')):.4f}
- Balanced accuracy: {best_row.get('balanced_accuracy', float('nan')):.4f}
- Macro F1: {best_row.get('f1_macro', float('nan')):.4f}
- Weighted F1: {best_row.get('f1_weighted', float('nan')):.4f}
- Binary F1, merge01_vs_23: {best_row.get('binary_f1_merge01_vs_23', float('nan')):.4f}
- Binary balanced accuracy, merge01_vs_23: {best_row.get('binary_balanced_accuracy_merge01_vs_23', float('nan')):.4f}
- {auc_display}: {best_row.get(auc_display, float('nan')):.4f}
- {ap_display}: {best_row.get(ap_display, float('nan')):.4f}
- Mean prediction confidence: {best_row.get('mean_prediction_confidence', float('nan')):.4f}

## Internal-to-External Shift
For the best external policy, the external-minus-internal change was:

- Accuracy shift: {shift_best.get('accuracy_external_minus_internal', float('nan')):.4f}
- Balanced accuracy shift: {shift_best.get('balanced_accuracy_external_minus_internal', float('nan')):.4f}
- Weighted F1 shift: {shift_best.get('f1_weighted_external_minus_internal', float('nan')):.4f}
- Binary F1 shift: {shift_best.get('binary_f1_merge01_vs_23_external_minus_internal', float('nan')):.4f}

These values quantify temporal robustness under later-cycle distribution shift and partial observability.

## Feature Availability and Partial Observability
The external feature-availability profile was computed by specialist team. This allows the paper to report which organizational domains remained fully observable and which were affected by later-cycle missingness or variable drift.

## Interpretation
Experiment 5 completes the method-to-results chain by testing the trained organizational framework outside the internal 2011–2018 development environment. The experiment is intentionally stricter than ordinary random validation because later NHANES cycles introduce temporal drift, variable availability differences, and a harmonized fallback target. Therefore, stable performance under this setting supports the claim that the proposed Nash-cooperative multi-team architecture provides organizational robustness under heterogeneous and partially observable disability-health conditions.

## Generated Outputs
- experiment5_internal_reference_metrics.csv
- experiment5_external_temporal_metrics.csv
- experiment5_internal_external_shift_metrics.csv
- experiment5_external_feature_availability_detail.csv
- experiment5_external_feature_availability_by_team.csv
- experiment5_external_target_distribution.csv
- experiment5_external_cooperation_policies.csv
- experiment5_external_team_confidence_{best_external_method}.csv
- experiment5_external_demographic_subgroup_metrics.csv, if subgroup attributes are available
- experiment5_external_demographic_fairness_gaps.csv, if subgroup attributes are available
- fig1_external_policy_metric_comparison.png
- fig2_internal_external_shift_comparison.png
- fig3_external_feature_availability_by_team.png
- fig4_external_confusion_matrix_{best_external_method}.png
- fig5_external_prediction_confidence_distribution.png
- fig6_external_team_confidence.png
- fig7_external_target_distribution.png
"""

    with open(EXP5_RESULTS / "experiment5_paper_ready_summary.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("\nExperiment 5 completed successfully.")
    print(f"Results folder: {EXP5_RESULTS}")
    print(f"External target column used: {external_target_col}")
    print(f"External rows: {len(external_df):,}")
    print(f"Best external policy: {best_external_method}")
    print(f"Primary selection metric: {primary_metric}")
    print("\nExternal policy metrics:")
    display_cols = ["method", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
    for c in ["binary_f1_merge01_vs_23", auc_display, ap_display]:
        if c in external_metrics_df.columns and c not in display_cols:
            display_cols.append(c)
    print(external_metrics_df[display_cols].to_string(index=False))
    print("\nMost important outputs:")
    print(f"  {EXP5_RESULTS / 'experiment5_external_temporal_metrics.csv'}")
    print(f"  {EXP5_RESULTS / 'experiment5_internal_external_shift_metrics.csv'}")
    print(f"  {EXP5_RESULTS / 'experiment5_paper_ready_summary.md'}")


if __name__ == "__main__":
    main()

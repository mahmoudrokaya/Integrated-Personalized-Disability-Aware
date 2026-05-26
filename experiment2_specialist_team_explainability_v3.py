# -*- coding: utf-8 -*-
"""
Experiment 2: Specialist-Team Contribution and Organizational Explainability
Version 2

This version is designed to be scientifically consistent with Experiment 1 v2.
It does NOT re-infer the architecture independently. Instead, it loads the
exact saved checkpoint metadata from Experiment 1, including:

- team names
- hidden dimension
- number of learners per team
- dropout
- target column
- feature order
- domain map
- domain indices
- class mapping
- scaler mean and scale
- categorical encoders / medians when available

This prevents invalid explainability caused by architecture mismatch.
"""

import os
import json
import random
import platform
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 0. Reproducibility and paths
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
EXP2_RESULTS = ROOT / "Experiment2" / "Results"

TRAIN_FILE = MODELING_DIR / "train_2011_2018.csv"
VAL_FILE = MODELING_DIR / "val_2011_2018.csv"
TEST_FILE = MODELING_DIR / "test_2011_2018.csv"
BEST_MODEL_FILE = EXP1_RESULTS / "experiment1_best_model.pt"

EXP2_RESULTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------
def save_json(obj, path: Path):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
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


def load_experiment1_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment 1 checkpoint not found:\n{path}\n"
            "Run experiment1_organizational_training_reporting_v2.py first."
        )

    checkpoint = torch.load(path, map_location=DEVICE)

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dictionary. Please rerun Experiment 1 v2.")

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain model_state_dict. "
            "Please rerun experiment1_organizational_training_reporting_v2.py."
        )

    return checkpoint


def extract_metadata(checkpoint: dict) -> dict:
    """
    Robust metadata extraction. Supports several possible key names used in the
    updated Experiment 1 saving logic.
    """
    config = get_first_existing(checkpoint, ["config", "CONFIG", "training_config"], {})

    team_names = get_first_existing(
        checkpoint,
        ["team_names", "specialist_team_names"],
        config.get("team_names") if isinstance(config, dict) else None,
    )

    domain_map = get_first_existing(
        checkpoint,
        ["domain_map", "domain_features", "team_feature_map"],
        None,
    )

    feature_columns = get_first_existing(
        checkpoint,
        ["feature_columns", "feature_cols", "features"],
        None,
    )

    target_col = get_first_existing(
        checkpoint,
        ["target_column", "target_col"],
        config.get("target_column") if isinstance(config, dict) else None,
    )

    class_mapping = get_first_existing(
        checkpoint,
        ["class_mapping", "class_to_idx", "target_mapping"],
        None,
    )

    num_classes = get_first_existing(
        checkpoint,
        ["num_classes"],
        config.get("num_classes") if isinstance(config, dict) else None,
    )

    hidden_dim = get_first_existing(
        checkpoint,
        ["hidden_dim"],
        config.get("hidden_dim") if isinstance(config, dict) else None,
    )

    n_learners = get_first_existing(
        checkpoint,
        ["num_learners_per_team", "n_learners", "learners_per_team"],
        config.get("num_learners_per_team") if isinstance(config, dict) else None,
    )

    dropout = get_first_existing(
        checkpoint,
        ["dropout"],
        config.get("dropout") if isinstance(config, dict) else 0.10,
    )

    scaler_mean = get_first_existing(checkpoint, ["scaler_mean", "feature_mean", "means"], None)
    scaler_scale = get_first_existing(checkpoint, ["scaler_scale", "feature_scale", "stds"], None)
    train_medians = get_first_existing(checkpoint, ["train_medians", "medians"], None)
    categorical_encoders = get_first_existing(checkpoint, ["categorical_encoders", "category_maps"], {})

    # Some checkpoints save all metadata under one sub-dictionary.
    meta = get_first_existing(checkpoint, ["metadata", "data_metadata", "checkpoint_manifest"], None)
    if isinstance(meta, dict):
        team_names = team_names or meta.get("team_names")
        domain_map = domain_map or meta.get("domain_map") or meta.get("domain_features")
        feature_columns = feature_columns or meta.get("feature_columns") or meta.get("feature_cols")
        target_col = target_col or meta.get("target_column") or meta.get("target_col")
        class_mapping = class_mapping or meta.get("class_mapping") or meta.get("class_to_idx")
        num_classes = num_classes or meta.get("num_classes")
        hidden_dim = hidden_dim or meta.get("hidden_dim")
        n_learners = n_learners or meta.get("num_learners_per_team") or meta.get("n_learners")
        dropout = dropout if dropout is not None else meta.get("dropout", 0.10)
        scaler_mean = scaler_mean or meta.get("scaler_mean") or meta.get("feature_mean")
        scaler_scale = scaler_scale or meta.get("scaler_scale") or meta.get("feature_scale")
        train_medians = train_medians or meta.get("train_medians")
        categorical_encoders = categorical_encoders or meta.get("categorical_encoders", {})

    # If metadata still lacks hidden size, infer from checkpoint weights.
    state = checkpoint["model_state_dict"]
    if hidden_dim is None:
        first_weight = None
        for k, v in state.items():
            if k.endswith("net.0.weight"):
                first_weight = v
                break
        if first_weight is not None:
            hidden_dim = int(first_weight.shape[0])

    if n_learners is None:
        # Count learners in the first team from state dict.
        learner_ids = set()
        for k in state.keys():
            if ".learners." in k:
                part = k.split(".learners.")[1].split(".")[0]
                if part.isdigit():
                    learner_ids.add(int(part))
        n_learners = max(learner_ids) + 1 if learner_ids else 3

    if team_names is None:
        # Infer from state keys: teams.<team>.learners...
        teams = []
        for k in state.keys():
            if k.startswith("teams."):
                parts = k.split(".")
                if len(parts) > 1:
                    teams.append(parts[1])
        team_names = sorted(list(set(teams)))

    if domain_map is None:
        raise ValueError(
            "The checkpoint does not contain domain_map/domain_features. "
            "Please rerun Experiment 1 v2 and ensure it saves domain_map."
        )

    if feature_columns is None:
        # reconstruct from domain map order if needed
        feature_columns = []
        for t in team_names:
            for col in domain_map[t]:
                if col not in feature_columns:
                    feature_columns.append(col)

    if target_col is None:
        # fallback used by earlier code
        target_col = "disability_stage"

    if num_classes is None:
        # infer from final layer output dimension
        for k, v in state.items():
            if k.endswith("net.3.weight"):
                num_classes = int(v.shape[0])
                break

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
        "dropout": float(dropout) if dropout is not None else 0.10,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "train_medians": train_medians,
        "categorical_encoders": categorical_encoders,
    }

    return metadata


def apply_experiment1_preprocessing(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Applies exactly the saved feature order and preprocessing references.
    This avoids re-deriving the model input space.
    """
    out = df.copy()
    feature_columns = metadata["feature_columns"]
    target_col = metadata["target_column"]

    # Ensure all expected columns exist.
    for col in feature_columns:
        if col not in out.columns:
            out[col] = np.nan

    # Categorical handling using saved encoders where available.
    cat_maps = metadata.get("categorical_encoders", {}) or {}
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            if col in cat_maps and isinstance(cat_maps[col], dict):
                out[col] = out[col].astype(str).map(cat_maps[col]).fillna(-1)
            else:
                out[col] = pd.factorize(out[col].astype(str))[0]

    # Fill missing values using saved medians if available.
    medians = metadata.get("train_medians", None)
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

    # Apply saved scaler when available.
    mean = metadata.get("scaler_mean", None)
    scale = metadata.get("scaler_scale", None)

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
        # Fallback only if old checkpoint lacks scaler metadata.
        vals = out[feature_columns]
        out[feature_columns] = (vals - vals.mean()) / vals.std().replace(0, 1.0)

    out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], 0).fillna(0)

    # Target mapping.
    if target_col not in out.columns:
        raise ValueError(f"Target column '{target_col}' is missing from test file.")

    class_mapping = metadata.get("class_mapping", None)
    if isinstance(class_mapping, dict):
        mapping = {}
        for k, v in class_mapping.items():
            mapping[k] = int(v)
            try:
                mapping[float(k)] = int(v)
            except Exception:
                pass
            try:
                mapping[int(float(k))] = int(v)
            except Exception:
                pass
        out[target_col] = out[target_col].map(mapping)
        if out[target_col].isna().any():
            # fallback if values are already encoded
            out[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    else:
        out[target_col] = pd.to_numeric(out[target_col], errors="coerce")

    out[target_col] = out[target_col].astype(int)

    return out


# ---------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------
class MultidomainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, metadata: dict):
        self.target_col = metadata["target_column"]
        self.team_names = metadata["team_names"]
        self.domain_map = metadata["domain_map"]
        self.y = torch.tensor(df[self.target_col].values, dtype=torch.long)

        self.domain_data = {}
        for team in self.team_names:
            cols = self.domain_map[team]
            self.domain_data[team] = torch.tensor(df[cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = {team: self.domain_data[team][idx] for team in self.team_names}
        return x, self.y[idx]


# ---------------------------------------------------------------------
# 3. Model architecture, exactly compatible with Experiment 1
# ---------------------------------------------------------------------
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
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, n_learners: int, dropout: float):
        super().__init__()
        self.learners = nn.ModuleList([
            LightweightLearner(input_dim, hidden_dim, num_classes, dropout)
            for _ in range(n_learners)
        ])
        self.intra_team_logits = nn.Parameter(torch.zeros(n_learners))

    def forward(self, x):
        learner_outputs = torch.stack([learner(x) for learner in self.learners], dim=1)
        learner_weights = torch.softmax(self.intra_team_logits, dim=0)
        team_output = torch.sum(learner_outputs * learner_weights.view(1, -1, 1), dim=1)
        return team_output, learner_outputs, learner_weights


class NashCooperativeOrganizationalModel(nn.Module):
    def __init__(self, metadata: dict):
        super().__init__()
        self.team_names = metadata["team_names"]
        self.domain_map = metadata["domain_map"]
        hidden_dim = metadata["hidden_dim"]
        num_classes = metadata["num_classes"]
        n_learners = metadata["num_learners_per_team"]
        dropout = metadata["dropout"]

        self.teams = nn.ModuleDict({
            team: SpecialistTeam(
                input_dim=len(self.domain_map[team]),
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                n_learners=n_learners,
                dropout=dropout,
            )
            for team in self.team_names
        })
        self.team_logits = nn.Parameter(torch.zeros(len(self.team_names)))

    def forward(self, x_dict, mask_team: Optional[str] = None, only_team: Optional[str] = None):
        team_outputs = []
        raw_team_dict = {}
        learner_weight_dict = {}

        for team in self.team_names:
            out, learner_outputs, learner_weights = self.teams[team](x_dict[team])
            team_outputs.append(out)
            raw_team_dict[team] = out
            learner_weight_dict[team] = learner_weights

        stacked = torch.stack(team_outputs, dim=1)
        weights = torch.softmax(self.team_logits, dim=0)

        if only_team is not None:
            adjusted = torch.zeros_like(weights)
            adjusted[self.team_names.index(only_team)] = 1.0
            weights = adjusted

        if mask_team is not None:
            adjusted = weights.clone()
            adjusted[self.team_names.index(mask_team)] = 0.0
            adjusted = adjusted / (adjusted.sum() + 1e-8)
            weights = adjusted

        logits = torch.sum(stacked * weights.view(1, -1, 1), dim=1)
        return logits, stacked, weights, raw_team_dict, learner_weight_dict


# ---------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------
@torch.no_grad()
def predict_model(model, loader, team_names, mask_team=None, only_team=None):
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    all_team_probs = {team: [] for team in team_names}
    losses = []
    n_items = 0

    for x_dict, y in loader:
        x_dict = {k: v.to(DEVICE) for k, v in x_dict.items()}
        y = y.to(DEVICE)

        logits, stacked, weights, raw_team_dict, _ = model(
            x_dict, mask_team=mask_team, only_team=only_team
        )
        loss = F.cross_entropy(logits, y)

        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

        losses.append(loss.item() * y.size(0))
        n_items += y.size(0)

        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())

        for team in team_names:
            all_team_probs[team].append(torch.softmax(raw_team_dict[team], dim=1).cpu().numpy())

    return {
        "loss": float(np.sum(losses) / max(n_items, 1)),
        "y_true": np.concatenate(all_y),
        "y_pred": np.concatenate(all_pred),
        "y_prob": np.concatenate(all_prob),
        "team_probs": {team: np.concatenate(parts) for team, parts in all_team_probs.items()},
    }


def compute_metrics(y_true, y_pred, y_prob):
    out = {
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
            out["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            out["average_precision"] = average_precision_score(y_true, y_prob[:, 1])
        else:
            out["roc_auc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
            y_bin = label_binarize(y_true, classes=list(range(y_prob.shape[1])))
            out["average_precision_macro"] = average_precision_score(y_bin, y_prob, average="macro")
    except Exception:
        pass

    return out


def evaluate_setting(model, loader, team_names, setting, mask_team=None, only_team=None):
    result = predict_model(model, loader, team_names, mask_team=mask_team, only_team=only_team)
    metrics = compute_metrics(result["y_true"], result["y_pred"], result["y_prob"])
    metrics["loss"] = result["loss"]
    metrics["setting"] = setting
    return result, metrics


# ---------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------
def plot_bar(df, x, y, title, ylabel, path, rotation=30):
    plt.figure(figsize=(9, 6))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_grouped_metrics(df, metric_cols, label_col, title, path):
    x = np.arange(len(df))
    width = 0.8 / max(len(metric_cols), 1)

    plt.figure(figsize=(11, 6))
    for i, metric in enumerate(metric_cols):
        if metric in df.columns:
            plt.bar(x + i * width, df[metric], width=width, label=metric)

    plt.xticks(x + width * (len(metric_cols) - 1) / 2, df[label_col].astype(str), rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion(cm, path, title="Confusion Matrix"):
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


def plot_confidence_distribution(y_true, y_pred, y_prob, path):
    conf = np.max(y_prob, axis=1)
    correct = y_true == y_pred

    plt.figure(figsize=(9, 6))
    plt.hist(conf[correct], bins=25, alpha=0.7, label="Correct")
    plt.hist(conf[~correct], bins=25, alpha=0.7, label="Incorrect")
    plt.xlabel("Prediction confidence")
    plt.ylabel("Frequency")
    plt.title("Prediction Confidence Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_team_signal_correlation(team_probs, path):
    team_names = list(team_probs.keys())
    vectors = []
    for team in team_names:
        vectors.append(team_probs[team].reshape(team_probs[team].shape[0], -1).mean(axis=1))
    mat = np.corrcoef(np.vstack(vectors))

    plt.figure(figsize=(8, 7))
    plt.imshow(mat, interpolation="nearest", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(len(team_names)), team_names, rotation=30, ha="right")
    plt.yticks(np.arange(len(team_names)), team_names)
    plt.title("Specialist-Team Prediction Signal Correlation")
    for i in range(len(team_names)):
        for j in range(len(team_names)):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 6. Explainability analyses
# ---------------------------------------------------------------------
def team_prediction_agreement(base_pred, team_probs):
    rows = []
    for team, prob in team_probs.items():
        team_pred = prob.argmax(axis=1)
        rows.append({
            "team": team,
            "agreement_with_global_prediction": accuracy_score(base_pred, team_pred),
            "mean_team_confidence": float(np.max(prob, axis=1).mean()),
        })
    return pd.DataFrame(rows)


def team_error_overlap(y_true, base_pred, team_probs):
    rows = []
    global_error = base_pred != y_true
    for team, prob in team_probs.items():
        team_pred = prob.argmax(axis=1)
        team_error = team_pred != y_true
        rows.append({
            "team": team,
            "team_error_rate": float(team_error.mean()),
            "global_error_rate": float(global_error.mean()),
            "shared_error_rate": float(np.logical_and(team_error, global_error).mean()),
            "team_only_error_rate": float(np.logical_and(team_error, ~global_error).mean()),
            "global_only_error_rate": float(np.logical_and(~team_error, global_error).mean()),
        })
    return pd.DataFrame(rows)


def subgroup_metrics(test_df, metadata, y_true, y_pred, y_prob):
    feature_cols = metadata["feature_columns"]
    candidates = [
        c for c in feature_cols
        if any(k in c.lower() for k in ["age", "sex", "gender", "race", "ethnicity", "education", "income", "poverty"])
    ]

    rows = []
    for col in candidates:
        values = test_df[col]
        if pd.api.types.is_numeric_dtype(values) and values.nunique() > 6:
            try:
                groups = pd.qcut(values, q=4, duplicates="drop").astype(str)
            except Exception:
                groups = values.astype(str)
        else:
            groups = values.astype(str)

        for group in sorted(groups.unique()):
            idx = (groups == group).values
            if idx.sum() < 20:
                continue
            met = compute_metrics(y_true[idx], y_pred[idx], y_prob[idx])
            rows.append({
                "attribute": col,
                "subgroup": group,
                "n": int(idx.sum()),
                "accuracy": met.get("accuracy", np.nan),
                "balanced_accuracy": met.get("balanced_accuracy", np.nan),
                "f1_macro": met.get("f1_macro", np.nan),
                "f1_weighted": met.get("f1_weighted", np.nan),
                "mean_prediction_confidence": met.get("mean_prediction_confidence", np.nan),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------
def main():
    print("Starting Experiment 2 v2: Specialist-Team Contribution and Organizational Explainability")
    print(f"Device: {DEVICE}")
    print(f"Output folder: {EXP2_RESULTS}")

    checkpoint = load_experiment1_checkpoint(BEST_MODEL_FILE)
    metadata = extract_metadata(checkpoint)

    save_json(metadata, EXP2_RESULTS / "loaded_experiment1_architecture_metadata.json")

    raw_test_df = pd.read_csv(TEST_FILE)
    test_df = apply_experiment1_preprocessing(raw_test_df, metadata)

    test_dataset = MultidomainDataset(test_df, metadata)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = NashCooperativeOrganizationalModel(metadata).to(DEVICE)

    # Strict loading is preferred now because architecture must match exactly.
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    team_names = metadata["team_names"]

    # Full model.
    base_result, base_metrics = evaluate_setting(
        model, test_loader, team_names, "full_cooperative_model"
    )

    save_json(base_metrics, EXP2_RESULTS / "full_model_metrics.json")
    pd.DataFrame([base_metrics]).to_csv(EXP2_RESULTS / "full_model_metrics.csv", index=False, encoding="utf-8-sig")

    report = classification_report(
        base_result["y_true"], base_result["y_pred"], output_dict=True, zero_division=0
    )
    save_json(report, EXP2_RESULTS / "classification_report_full_model.json")
    pd.DataFrame(report).transpose().to_csv(
        EXP2_RESULTS / "classification_report_full_model.csv", encoding="utf-8-sig"
    )

    cm = confusion_matrix(base_result["y_true"], base_result["y_pred"])
    pd.DataFrame(cm).to_csv(EXP2_RESULTS / "confusion_matrix_full_model.csv", index=False)
    plot_confusion(cm, EXP2_RESULTS / "fig1_confusion_matrix_full_model.png")

    # Final cooperative weights.
    with torch.no_grad():
        weights = torch.softmax(model.team_logits, dim=0).cpu().numpy()

    team_weights_df = pd.DataFrame({
        "team": team_names,
        "cooperative_weight": weights,
        "num_features": [len(metadata["domain_map"][t]) for t in team_names],
        "features": ["; ".join(metadata["domain_map"][t]) for t in team_names],
    })
    save_dataframe(team_weights_df, EXP2_RESULTS / "team_cooperative_weights.csv")
    plot_bar(
        team_weights_df, "team", "cooperative_weight",
        "Final Nash-Cooperative Specialist-Team Weights",
        "Cooperative weight",
        EXP2_RESULTS / "fig2_final_team_weights.png"
    )

    # Single-team-only performance.
    single_rows = []
    for team in team_names:
        _, met = evaluate_setting(model, test_loader, team_names, f"only_{team}", only_team=team)
        met["team"] = team
        single_rows.append(met)
    single_df = pd.DataFrame(single_rows)
    save_dataframe(single_df, EXP2_RESULTS / "single_team_only_metrics.csv")
    plot_grouped_metrics(
        single_df,
        ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"],
        "team",
        "Single-Team-Only Performance",
        EXP2_RESULTS / "fig3_single_team_only_performance.png"
    )

    # Leave-one-team-out.
    ablation_rows = []
    for team in team_names:
        _, met = evaluate_setting(model, test_loader, team_names, f"without_{team}", mask_team=team)
        met["removed_team"] = team
        ablation_rows.append(met)

    ablation_df = pd.DataFrame(ablation_rows)
    for metric in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "average_precision", "roc_auc_ovr"]:
        if metric in ablation_df.columns and metric in base_metrics:
            ablation_df[f"{metric}_drop_vs_full"] = base_metrics[metric] - ablation_df[metric]

    save_dataframe(ablation_df, EXP2_RESULTS / "leave_one_team_out_ablation_metrics.csv")
    drop_cols = [c for c in ["accuracy_drop_vs_full", "f1_weighted_drop_vs_full", "roc_auc_drop_vs_full", "roc_auc_ovr_drop_vs_full"] if c in ablation_df.columns]
    if drop_cols:
        plot_grouped_metrics(
            ablation_df,
            drop_cols,
            "removed_team",
            "Performance Drop After Removing Each Specialist Team",
            EXP2_RESULTS / "fig4_leave_one_team_out_drops.png"
        )

    # Agreement and error overlap.
    agreement_df = team_prediction_agreement(base_result["y_pred"], base_result["team_probs"])
    save_dataframe(agreement_df, EXP2_RESULTS / "team_prediction_agreement.csv")
    plot_bar(
        agreement_df,
        "team",
        "agreement_with_global_prediction",
        "Agreement Between Specialist Teams and Global Prediction",
        "Agreement",
        EXP2_RESULTS / "fig5_team_global_agreement.png"
    )

    error_df = team_error_overlap(base_result["y_true"], base_result["y_pred"], base_result["team_probs"])
    save_dataframe(error_df, EXP2_RESULTS / "team_error_overlap.csv")
    plot_grouped_metrics(
        error_df,
        ["team_error_rate", "shared_error_rate", "team_only_error_rate"],
        "team",
        "Specialist-Team Error Overlap Analysis",
        EXP2_RESULTS / "fig6_team_error_overlap.png"
    )

    plot_team_signal_correlation(
        base_result["team_probs"],
        EXP2_RESULTS / "fig7_team_prediction_signal_correlation.png"
    )
    plot_confidence_distribution(
        base_result["y_true"],
        base_result["y_pred"],
        base_result["y_prob"],
        EXP2_RESULTS / "fig8_prediction_confidence_distribution.png"
    )

    # Domain permutation importance.
    permutation_rows = []
    for team in team_names:
        shuffled_df = test_df.copy()
        for col in metadata["domain_map"][team]:
            shuffled_df[col] = np.random.permutation(shuffled_df[col].values)

        shuffled_dataset = MultidomainDataset(shuffled_df, metadata)
        shuffled_loader = DataLoader(shuffled_dataset, batch_size=128, shuffle=False)
        _, met = evaluate_setting(model, shuffled_loader, team_names, f"permuted_{team}")

        row = {
            "permuted_domain": team,
            "accuracy": met.get("accuracy", np.nan),
            "balanced_accuracy": met.get("balanced_accuracy", np.nan),
            "f1_macro": met.get("f1_macro", np.nan),
            "f1_weighted": met.get("f1_weighted", np.nan),
            "roc_auc": met.get("roc_auc", np.nan),
            "average_precision": met.get("average_precision", np.nan),
        }
        for metric in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "average_precision"]:
            if metric in base_metrics:
                row[f"{metric}_drop"] = base_metrics[metric] - row.get(metric, np.nan)
        permutation_rows.append(row)

    permutation_df = pd.DataFrame(permutation_rows)
    save_dataframe(permutation_df, EXP2_RESULTS / "domain_permutation_importance.csv")
    perm_drop_cols = [c for c in ["accuracy_drop", "f1_weighted_drop", "roc_auc_drop", "average_precision_drop"] if c in permutation_df.columns]
    if perm_drop_cols:
        plot_grouped_metrics(
            permutation_df,
            perm_drop_cols,
            "permuted_domain",
            "Domain-Level Permutation Importance",
            EXP2_RESULTS / "fig9_domain_permutation_importance.png"
        )

    # Fairness subgroup analysis.
    fairness_df = subgroup_metrics(
        test_df,
        metadata,
        base_result["y_true"],
        base_result["y_pred"],
        base_result["y_prob"],
    )

    if not fairness_df.empty:
        save_dataframe(fairness_df, EXP2_RESULTS / "demographic_subgroup_performance.csv")
        disparity_rows = []
        for attr, g in fairness_df.groupby("attribute"):
            disparity_rows.append({
                "attribute": attr,
                "num_subgroups": g["subgroup"].nunique(),
                "min_f1_macro": g["f1_macro"].min(),
                "max_f1_macro": g["f1_macro"].max(),
                "f1_macro_gap": g["f1_macro"].max() - g["f1_macro"].min(),
                "min_balanced_accuracy": g["balanced_accuracy"].min(),
                "max_balanced_accuracy": g["balanced_accuracy"].max(),
                "balanced_accuracy_gap": g["balanced_accuracy"].max() - g["balanced_accuracy"].min(),
            })
        disparity_df = pd.DataFrame(disparity_rows)
        save_dataframe(disparity_df, EXP2_RESULTS / "demographic_fairness_gaps.csv")
        plot_bar(
            disparity_df,
            "attribute",
            "f1_macro_gap",
            "Subgroup F1-Macro Gap by Demographic Attribute",
            "F1-macro gap",
            EXP2_RESULTS / "fig10_demographic_f1_gap.png"
        )

    # Integrated explainability summary.
    summary_rows = []
    for team in team_names:
        weight_row = team_weights_df[team_weights_df["team"] == team].iloc[0]
        single_row = single_df[single_df["team"] == team].iloc[0]
        ablation_row = ablation_df[ablation_df["removed_team"] == team].iloc[0]
        perm_row = permutation_df[permutation_df["permuted_domain"] == team].iloc[0]
        agreement_row = agreement_df[agreement_df["team"] == team].iloc[0]

        summary_rows.append({
            "team": team,
            "cooperative_weight": weight_row["cooperative_weight"],
            "num_features": weight_row["num_features"],
            "single_team_accuracy": single_row.get("accuracy", np.nan),
            "single_team_f1_weighted": single_row.get("f1_weighted", np.nan),
            "single_team_roc_auc": single_row.get("roc_auc", np.nan),
            "drop_when_removed_accuracy": ablation_row.get("accuracy_drop_vs_full", np.nan),
            "drop_when_removed_f1_weighted": ablation_row.get("f1_weighted_drop_vs_full", np.nan),
            "drop_when_removed_roc_auc": ablation_row.get("roc_auc_drop_vs_full", np.nan),
            "permutation_accuracy_drop": perm_row.get("accuracy_drop", np.nan),
            "permutation_f1_weighted_drop": perm_row.get("f1_weighted_drop", np.nan),
            "permutation_roc_auc_drop": perm_row.get("roc_auc_drop", np.nan),
            "agreement_with_global_prediction": agreement_row.get("agreement_with_global_prediction", np.nan),
            "mean_team_confidence": agreement_row.get("mean_team_confidence", np.nan),
        })

    integrated_df = pd.DataFrame(summary_rows)
    save_dataframe(integrated_df, EXP2_RESULTS / "experiment2_integrated_team_explainability_summary.csv")

    # Paper-ready summary.
    top_weight = team_weights_df.sort_values("cooperative_weight", ascending=False).iloc[0]
    top_single = single_df.sort_values("f1_weighted", ascending=False).iloc[0]
    if "roc_auc_drop" in permutation_df.columns:
        top_perm = permutation_df.sort_values("roc_auc_drop", ascending=False).iloc[0]
        top_perm_metric = f"AUC drop = {top_perm.get('roc_auc_drop', np.nan):.4f}"
    else:
        top_perm = permutation_df.sort_values("f1_weighted_drop", ascending=False).iloc[0]
        top_perm_metric = f"weighted-F1 drop = {top_perm.get('f1_weighted_drop', np.nan):.4f}"

    if "roc_auc_drop_vs_full" in ablation_df.columns:
        top_ablation = ablation_df.sort_values("roc_auc_drop_vs_full", ascending=False).iloc[0]
        top_ablation_metric = f"AUC drop = {top_ablation.get('roc_auc_drop_vs_full', np.nan):.4f}"
    else:
        top_ablation = ablation_df.sort_values("f1_weighted_drop_vs_full", ascending=False).iloc[0]
        top_ablation_metric = f"weighted-F1 drop = {top_ablation.get('f1_weighted_drop_vs_full', np.nan):.4f}"

    report_md = f"""# Experiment 2: Specialist-Team Contribution and Organizational Explainability

## Objective
Experiment 2 evaluates the interpretability of the trained Nash-cooperative organizational model from Experiment 1. Unlike a separate training experiment, this analysis directly reloads the exact Experiment 1 checkpoint, architecture metadata, specialist-team decomposition, feature ordering, and scaling references. This ensures that all explainability outputs are scientifically consistent with the trained organizational framework.

## Full Cooperative Model Performance
The full cooperative model achieved:

- Accuracy: {base_metrics.get('accuracy', float('nan')):.4f}
- Balanced accuracy: {base_metrics.get('balanced_accuracy', float('nan')):.4f}
- Macro F1: {base_metrics.get('f1_macro', float('nan')):.4f}
- Weighted F1: {base_metrics.get('f1_weighted', float('nan')):.4f}
- ROC-AUC: {base_metrics.get('roc_auc', float('nan')):.4f}
- Average precision: {base_metrics.get('average_precision', float('nan')):.4f}
- Mean prediction confidence: {base_metrics.get('mean_prediction_confidence', float('nan')):.4f}

## Specialist-Team Contribution
The highest Nash-cooperative weight was assigned to the **{top_weight['team']}** team, with a final weight of {top_weight['cooperative_weight']:.4f}. This indicates that this specialist domain exerted the strongest learned organizational influence in the final cooperative prediction.

## Single-Team Explainability
The strongest single-team-only configuration was **{top_single['team']}**, achieving weighted F1 = {top_single.get('f1_weighted', float('nan')):.4f}. This measures the independent predictive capacity of each specialist population when isolated from the cooperative organization.

## Leave-One-Team-Out Contribution
The largest degradation after removing one team occurred for **{top_ablation['removed_team']}**, with {top_ablation_metric}. This provides direct evidence of the contribution of that specialist team to the full organizational model.

## Domain Permutation Importance
The strongest domain-level permutation effect was observed for **{top_perm['permuted_domain']}**, with {top_perm_metric}. This indicates that disturbing this domain caused the largest reduction in cooperative prediction quality.

## Generated Report Outputs
The script generated publication-ready tables and figures covering:

- final cooperative specialist-team weights
- single-team-only performance
- leave-one-team-out ablation
- domain-level permutation importance
- team-global prediction agreement
- team error overlap
- specialist-team signal correlation
- confidence distribution
- demographic subgroup performance
- demographic fairness gaps
- integrated team explainability summary

## Manuscript Interpretation
Experiment 2 confirms whether the proposed model operates as an interpretable organizational intelligence system. The analysis links final prediction behavior to specialist-team influence, domain-level sensitivity, team agreement, ablation-based necessity, and demographic consistency. These outputs directly support the Methods section by empirically validating the specialist-team explainability mechanism and the Nash-cooperative adaptive weighting strategy.
"""

    with open(EXP2_RESULTS / "experiment2_paper_ready_summary.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    # Runtime metadata.
    runtime = {
        "experiment": "Experiment 2 v2",
        "device": str(DEVICE),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "input_checkpoint": str(BEST_MODEL_FILE),
        "test_file": str(TEST_FILE),
        "output_folder": str(EXP2_RESULTS),
        "loaded_hidden_dim": metadata["hidden_dim"],
        "loaded_num_learners_per_team": metadata["num_learners_per_team"],
        "loaded_num_classes": metadata["num_classes"],
        "loaded_team_names": metadata["team_names"],
    }
    save_json(runtime, EXP2_RESULTS / "experiment2_runtime_metadata.json")

    print("\nExperiment 2 v2 completed successfully.")
    print(f"Results folder: {EXP2_RESULTS}")
    print("\nFull cooperative model metrics:")
    for k, v in base_metrics.items():
        if isinstance(v, (float, int)):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print("\nFinal specialist-team weights:")
    for _, row in team_weights_df.iterrows():
        print(f"  {row['team']}: {row['cooperative_weight']:.4f}")

    print("\nMost important outputs:")
    print(f"  {EXP2_RESULTS / 'experiment2_integrated_team_explainability_summary.csv'}")
    print(f"  {EXP2_RESULTS / 'experiment2_paper_ready_summary.md'}")


if __name__ == "__main__":
    main()

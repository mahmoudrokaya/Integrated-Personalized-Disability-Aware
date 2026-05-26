# -*- coding: utf-8 -*-
"""
Experiment 3: Swarm Optimization and Cooperative Adaptation
Paper: Nash-Cooperative Swarm Intelligence for Lightweight Edge Learning
       Through Multi-Team Adaptive Optimization

Scientific purpose
------------------
Experiment 3 evaluates whether swarm-guided cooperative adaptation improves
organizational learning beyond static or purely gradient-based cooperation.

This experiment is aligned with Methods Section 3 by testing:
1. Static equal cooperation
2. Learned Nash-cooperative weighting from Experiment 1
3. Swarm-guided cooperative adaptation using:
   - PSO-style adaptation
   - ACO-style adaptation
   - BCO-style adaptation
   - HHO-style adaptation
   - Hybrid swarm adaptation

The script reloads the exact Experiment 1 checkpoint metadata, architecture,
feature ordering, domain map, and preprocessing references. It then evaluates
alternative cooperation-weight policies without changing the specialist-team
feature decomposition.

Outputs
-------
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/Experiment3/Results

Report-ready outputs include:
- comparison metrics table
- cooperative weight table
- swarm convergence traces
- ROC/PR-oriented metrics where available
- adaptation stability summaries
- publication-ready plots
- Markdown summary for manuscript writing
"""

import json
import random
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

TRAIN_FILE = MODELING_DIR / "train_2011_2018.csv"
VAL_FILE = MODELING_DIR / "val_2011_2018.csv"
TEST_FILE = MODELING_DIR / "test_2011_2018.csv"
BEST_MODEL_FILE = EXP1_RESULTS / "experiment1_best_model.pt"

EXP3_RESULTS.mkdir(parents=True, exist_ok=True)

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


# ---------------------------------------------------------------------
# 2. Load Experiment 1 metadata exactly
# ---------------------------------------------------------------------
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


def extract_metadata(checkpoint: dict) -> dict:
    config = get_first_existing(checkpoint, ["config", "CONFIG", "training_config"], {})

    meta = get_first_existing(checkpoint, ["metadata", "data_metadata", "checkpoint_manifest"], {})
    if meta is None:
        meta = {}

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
        raise ValueError("Checkpoint does not include domain_map. Rerun Experiment 1 v2.")

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


def apply_preprocessing(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    out = df.copy()
    feature_columns = metadata["feature_columns"]
    target_col = metadata["target_column"]

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

    class_mapping = metadata.get("class_mapping")
    if isinstance(class_mapping, dict):
        mapping = {}
        for k, v in class_mapping.items():
            mapping[k] = int(v)
            try:
                mapping[int(float(k))] = int(v)
            except Exception:
                pass
            try:
                mapping[float(k)] = int(v)
            except Exception:
                pass
        mapped = out[target_col].map(mapping)
        if mapped.isna().any():
            mapped = pd.to_numeric(out[target_col], errors="coerce")
        out[target_col] = mapped.astype(int)
    else:
        out[target_col] = pd.to_numeric(out[target_col], errors="coerce").astype(int)

    return out


# ---------------------------------------------------------------------
# 3. Dataset and model
# ---------------------------------------------------------------------
class MultidomainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, metadata: dict):
        self.target_col = metadata["target_column"]
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
        # Must match Experiment 1 checkpoint parameter name.
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

    def forward(self, x_dict):
        team_outputs = []
        raw_team_dict = {}
        for team in self.team_names:
            out, _, _ = self.teams[team](x_dict[team])
            team_outputs.append(out)
            raw_team_dict[team] = out
        stacked = torch.stack(team_outputs, dim=1)
        learned_weights = torch.softmax(self.team_logits, dim=0)
        logits = torch.sum(stacked * learned_weights.view(1, -1, 1), dim=1)
        return logits, stacked, learned_weights, raw_team_dict


@torch.no_grad()
def collect_team_outputs(model, loader, metadata):
    model.eval()
    y_all = []
    stacked_all = []

    for x_dict, y in loader:
        x_dict = {k: v.to(DEVICE) for k, v in x_dict.items()}
        logits, stacked, weights, raw_team = model(x_dict)
        y_all.append(y.numpy())
        stacked_all.append(stacked.cpu().numpy())

    y_true = np.concatenate(y_all)
    stacked = np.concatenate(stacked_all, axis=0)  # [N, Teams, Classes]
    return y_true, stacked


# ---------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------
def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def evaluate_weight_vector(weights, y_true, stacked_outputs):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-12, None)
    weights = weights / weights.sum()

    logits = np.sum(stacked_outputs * weights.reshape(1, -1, 1), axis=1)
    prob = softmax_np(logits, axis=1)
    pred = np.argmax(prob, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "mean_prediction_confidence": float(np.max(prob, axis=1).mean()),
    }

    try:
        if prob.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, prob[:, 1])
            metrics["average_precision"] = average_precision_score(y_true, prob[:, 1])
        else:
            metrics["roc_auc_ovr"] = roc_auc_score(y_true, prob, multi_class="ovr")
            y_bin = label_binarize(y_true, classes=list(range(prob.shape[1])))
            metrics["average_precision_macro"] = average_precision_score(y_bin, prob, average="macro")
    except Exception:
        pass

    return metrics, pred, prob


def objective_score(metrics: dict):
    # Balanced objective aligned with the paper: performance + robustness preference.
    auc = metrics.get("roc_auc", metrics.get("roc_auc_ovr", 0.0))
    f1w = metrics.get("f1_weighted", 0.0)
    bal = metrics.get("balanced_accuracy", 0.0)
    f1m = metrics.get("f1_macro", 0.0)
    return 0.35 * auc + 0.30 * f1w + 0.20 * bal + 0.15 * f1m


def diversity_penalty(weights):
    # Penalize extreme collapse, but mildly.
    weights = np.asarray(weights)
    uniform = np.ones_like(weights) / len(weights)
    return float(np.sum((weights - uniform) ** 2))


def fitness(weights, y_true, stacked_outputs, diversity_lambda=0.03):
    metrics, _, _ = evaluate_weight_vector(weights, y_true, stacked_outputs)
    return objective_score(metrics) - diversity_lambda * diversity_penalty(weights)


# ---------------------------------------------------------------------
# 5. Swarm optimizers over cooperative team weights
# ---------------------------------------------------------------------
def normalize_simplex(v):
    v = np.asarray(v, dtype=np.float64)
    v = np.maximum(v, 1e-8)
    return v / v.sum()


def random_simplex(n):
    x = np.random.rand(n)
    return normalize_simplex(x)


def pso_optimize(y_true, stacked_outputs, n_particles=30, iterations=40):
    n_teams = stacked_outputs.shape[1]
    positions = np.array([random_simplex(n_teams) for _ in range(n_particles)])
    velocities = np.zeros_like(positions)

    pbest = positions.copy()
    pbest_scores = np.array([fitness(w, y_true, stacked_outputs) for w in pbest])
    gbest = pbest[np.argmax(pbest_scores)].copy()
    gbest_score = np.max(pbest_scores)

    trace = []
    for it in range(iterations):
        w_inertia = 0.72
        c1 = 1.35
        c2 = 1.35
        for i in range(n_particles):
            r1 = np.random.rand(n_teams)
            r2 = np.random.rand(n_teams)
            velocities[i] = (
                w_inertia * velocities[i]
                + c1 * r1 * (pbest[i] - positions[i])
                + c2 * r2 * (gbest - positions[i])
            )
            positions[i] = normalize_simplex(positions[i] + velocities[i])
            s = fitness(positions[i], y_true, stacked_outputs)
            if s > pbest_scores[i]:
                pbest[i] = positions[i].copy()
                pbest_scores[i] = s

        idx = np.argmax(pbest_scores)
        if pbest_scores[idx] > gbest_score:
            gbest = pbest[idx].copy()
            gbest_score = pbest_scores[idx]
        trace.append({"iteration": it + 1, "best_score": float(gbest_score)})

    return gbest, pd.DataFrame(trace)


def aco_optimize(y_true, stacked_outputs, n_ants=35, iterations=40):
    n_teams = stacked_outputs.shape[1]
    pheromone = np.ones(n_teams) / n_teams
    best = pheromone.copy()
    best_score = fitness(best, y_true, stacked_outputs)
    trace = []

    for it in range(iterations):
        candidates = []
        scores = []
        for _ in range(n_ants):
            concentration = 1.0 + 15.0 * pheromone
            w = np.random.dirichlet(concentration)
            s = fitness(w, y_true, stacked_outputs)
            candidates.append(w)
            scores.append(s)

        scores = np.array(scores)
        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best = candidates[idx].copy()
            best_score = float(scores[idx])

        # Evaporation and reinforcement.
        pheromone = 0.82 * pheromone + 0.18 * best
        pheromone = normalize_simplex(pheromone)
        trace.append({"iteration": it + 1, "best_score": float(best_score)})

    return best, pd.DataFrame(trace)


def bco_optimize(y_true, stacked_outputs, n_bees=35, iterations=40):
    n_teams = stacked_outputs.shape[1]
    population = np.array([random_simplex(n_teams) for _ in range(n_bees)])
    scores = np.array([fitness(w, y_true, stacked_outputs) for w in population])
    best = population[np.argmax(scores)].copy()
    best_score = float(np.max(scores))
    trace = []

    for it in range(iterations):
        elite_count = max(3, n_bees // 5)
        elite_idx = np.argsort(scores)[-elite_count:]
        new_pop = []

        # Exploit elite bees.
        for idx in elite_idx:
            center = population[idx]
            for _ in range(max(1, n_bees // elite_count // 2)):
                noise = np.random.normal(0, 0.08 * (1 - it / iterations), size=n_teams)
                new_pop.append(normalize_simplex(center + noise))

        # Explore random scouts.
        while len(new_pop) < n_bees:
            if np.random.rand() < 0.65:
                noise = np.random.normal(0, 0.12, size=n_teams)
                new_pop.append(normalize_simplex(best + noise))
            else:
                new_pop.append(random_simplex(n_teams))

        population = np.array(new_pop[:n_bees])
        scores = np.array([fitness(w, y_true, stacked_outputs) for w in population])

        if np.max(scores) > best_score:
            best = population[np.argmax(scores)].copy()
            best_score = float(np.max(scores))

        trace.append({"iteration": it + 1, "best_score": float(best_score)})

    return best, pd.DataFrame(trace)


def hho_optimize(y_true, stacked_outputs, n_hawks=35, iterations=40):
    n_teams = stacked_outputs.shape[1]
    hawks = np.array([random_simplex(n_teams) for _ in range(n_hawks)])
    scores = np.array([fitness(w, y_true, stacked_outputs) for w in hawks])
    rabbit = hawks[np.argmax(scores)].copy()
    rabbit_score = float(np.max(scores))
    trace = []

    for it in range(iterations):
        E1 = 2 * (1 - (it + 1) / iterations)
        mean_pos = np.mean(hawks, axis=0)

        for i in range(n_hawks):
            E0 = 2 * np.random.rand() - 1
            escaping_energy = E1 * E0

            if abs(escaping_energy) >= 1:
                # Exploration.
                rand_hawk = hawks[np.random.randint(n_hawks)]
                q = np.random.rand()
                if q < 0.5:
                    new_pos = rand_hawk - np.random.rand(n_teams) * abs(rand_hawk - 2 * np.random.rand(n_teams) * hawks[i])
                else:
                    new_pos = rabbit - mean_pos - np.random.rand(n_teams) * (np.random.rand(n_teams))
            else:
                # Exploitation around rabbit.
                jump_strength = 2 * (1 - np.random.rand())
                if np.random.rand() < 0.5:
                    new_pos = rabbit - escaping_energy * abs(jump_strength * rabbit - hawks[i])
                else:
                    levy_like = np.random.normal(0, 0.05, size=n_teams)
                    new_pos = rabbit - escaping_energy * abs(rabbit - hawks[i]) + levy_like

            new_pos = normalize_simplex(new_pos)
            new_score = fitness(new_pos, y_true, stacked_outputs)
            if new_score > scores[i]:
                hawks[i] = new_pos
                scores[i] = new_score

        if np.max(scores) > rabbit_score:
            rabbit = hawks[np.argmax(scores)].copy()
            rabbit_score = float(np.max(scores))

        trace.append({"iteration": it + 1, "best_score": float(rabbit_score)})

    return rabbit, pd.DataFrame(trace)


def hybrid_optimize(y_true, stacked_outputs):
    # Use all swarm mechanisms and select the best cooperative policy.
    candidates = []
    traces = []

    for name, fn in [
        ("PSO", pso_optimize),
        ("ACO", aco_optimize),
        ("BCO", bco_optimize),
        ("HHO", hho_optimize),
    ]:
        w, tr = fn(y_true, stacked_outputs)
        score = fitness(w, y_true, stacked_outputs)
        candidates.append((name, w, score))
        tr["optimizer"] = name
        traces.append(tr)

    best_name, best_w, best_score = sorted(candidates, key=lambda x: x[2], reverse=True)[0]
    trace_df = pd.concat(traces, ignore_index=True)
    return best_w, trace_df, best_name


# ---------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------
def plot_weight_comparison(weight_df, team_names, path):
    methods = weight_df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.8 / len(team_names)

    plt.figure(figsize=(12, 6))
    for i, team in enumerate(team_names):
        plt.bar(x + i * width, weight_df[team], width=width, label=team)

    plt.xticks(x + width * (len(team_names) - 1) / 2, methods, rotation=25, ha="right")
    plt.ylabel("Cooperative weight")
    plt.title("Cooperative Team Weights Across Adaptation Strategies")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_metric_comparison(metrics_df, path):
    cols = [c for c in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"] if c in metrics_df.columns]
    x = np.arange(len(metrics_df))
    width = 0.8 / len(cols)

    plt.figure(figsize=(11, 6))
    for i, col in enumerate(cols):
        plt.bar(x + i * width, metrics_df[col], width=width, label=col)
    plt.xticks(x + width * (len(cols) - 1) / 2, metrics_df["method"], rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title("Performance Comparison of Cooperative Adaptation Strategies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_auc_comparison(metrics_df, path):
    auc_col = "roc_auc" if "roc_auc" in metrics_df.columns else "roc_auc_ovr"
    if auc_col not in metrics_df.columns:
        return

    plt.figure(figsize=(9, 6))
    plt.bar(metrics_df["method"], metrics_df[auc_col])
    plt.ylabel(auc_col)
    plt.title("AUC Comparison of Cooperative Adaptation Strategies")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_convergence(trace_df, path):
    if trace_df.empty:
        return
    plt.figure(figsize=(10, 6))
    for opt, g in trace_df.groupby("optimizer"):
        plt.plot(g["iteration"], g["best_score"], label=opt)
    plt.xlabel("Iteration")
    plt.ylabel("Best objective score")
    plt.title("Swarm Adaptation Convergence")
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


# ---------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------
def main():
    print("Starting Experiment 3: Swarm Optimization and Cooperative Adaptation")
    print(f"Device: {DEVICE}")
    print(f"Output folder: {EXP3_RESULTS}")

    checkpoint = load_checkpoint(BEST_MODEL_FILE)
    metadata = extract_metadata(checkpoint)
    team_names = metadata["team_names"]

    save_json(metadata, EXP3_RESULTS / "loaded_experiment1_architecture_metadata.json")

    raw_val_df = pd.read_csv(VAL_FILE)
    raw_test_df = pd.read_csv(TEST_FILE)

    val_df = apply_preprocessing(raw_val_df, metadata)
    test_df = apply_preprocessing(raw_test_df, metadata)

    val_loader = DataLoader(MultidomainDataset(val_df, metadata), batch_size=256, shuffle=False)
    test_loader = DataLoader(MultidomainDataset(test_df, metadata), batch_size=256, shuffle=False)

    model = NashCooperativeOrganizationalModel(metadata).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    y_val, stacked_val = collect_team_outputs(model, val_loader, metadata)
    y_test, stacked_test = collect_team_outputs(model, test_loader, metadata)

    # Baseline cooperative policies.
    with torch.no_grad():
        learned_weights = torch.softmax(model.team_logits, dim=0).cpu().numpy()

    n_teams = len(team_names)
    uniform_weights = np.ones(n_teams) / n_teams

    # Swarm adaptation is fitted on validation set and evaluated on test set.
    pso_w, pso_trace = pso_optimize(y_val, stacked_val)
    pso_trace["optimizer"] = "PSO"

    aco_w, aco_trace = aco_optimize(y_val, stacked_val)
    aco_trace["optimizer"] = "ACO"

    bco_w, bco_trace = bco_optimize(y_val, stacked_val)
    bco_trace["optimizer"] = "BCO"

    hho_w, hho_trace = hho_optimize(y_val, stacked_val)
    hho_trace["optimizer"] = "HHO"

    hybrid_w, hybrid_trace, hybrid_source = hybrid_optimize(y_val, stacked_val)
    hybrid_trace["optimizer"] = "HybridComponents"

    policies = {
        "Static_Equal_Cooperation": uniform_weights,
        "Experiment1_Learned_Nash": learned_weights,
        "PSO_Adaptive_Cooperation": pso_w,
        "ACO_Adaptive_Cooperation": aco_w,
        "BCO_Adaptive_Cooperation": bco_w,
        "HHO_Adaptive_Cooperation": hho_w,
        "Hybrid_Swarm_Cooperation": hybrid_w,
    }

    # Evaluate all policies on validation and test.
    test_rows = []
    val_rows = []
    weight_rows = []

    predictions_for_best = {}

    for method, weights in policies.items():
        val_metrics, _, _ = evaluate_weight_vector(weights, y_val, stacked_val)
        test_metrics, test_pred, test_prob = evaluate_weight_vector(weights, y_test, stacked_test)

        val_metrics["method"] = method
        test_metrics["method"] = method
        val_metrics["objective_score"] = objective_score(val_metrics)
        test_metrics["objective_score"] = objective_score(test_metrics)
        val_metrics["diversity_penalty"] = diversity_penalty(weights)
        test_metrics["diversity_penalty"] = diversity_penalty(weights)

        val_rows.append(val_metrics)
        test_rows.append(test_metrics)

        row = {"method": method}
        for i, team in enumerate(team_names):
            row[team] = float(weights[i])
        row["weight_entropy"] = float(-np.sum(weights * np.log(weights + 1e-12)))
        row["max_weight"] = float(np.max(weights))
        row["min_weight"] = float(np.min(weights))
        row["hybrid_source"] = hybrid_source if method == "Hybrid_Swarm_Cooperation" else ""
        weight_rows.append(row)

        predictions_for_best[method] = (test_pred, test_prob)

    val_metrics_df = pd.DataFrame(val_rows)
    test_metrics_df = pd.DataFrame(test_rows)
    weights_df = pd.DataFrame(weight_rows)

    save_dataframe(val_metrics_df, EXP3_RESULTS / "experiment3_validation_metrics.csv")
    save_dataframe(test_metrics_df, EXP3_RESULTS / "experiment3_test_metrics.csv")
    save_dataframe(weights_df, EXP3_RESULTS / "experiment3_cooperative_weight_policies.csv")

    all_trace = pd.concat(
        [pso_trace, aco_trace, bco_trace, hho_trace],
        ignore_index=True,
    )
    save_dataframe(all_trace, EXP3_RESULTS / "experiment3_swarm_convergence_traces.csv")

    # Improvement table versus learned Nash and static equal.
    learned_row = test_metrics_df[test_metrics_df["method"] == "Experiment1_Learned_Nash"].iloc[0]
    static_row = test_metrics_df[test_metrics_df["method"] == "Static_Equal_Cooperation"].iloc[0]

    improvement_rows = []
    for _, row in test_metrics_df.iterrows():
        imp = {"method": row["method"]}
        for metric in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "roc_auc_ovr", "average_precision", "average_precision_macro", "objective_score"]:
            if metric in test_metrics_df.columns:
                imp[f"{metric}_gain_vs_learned_nash"] = row.get(metric, np.nan) - learned_row.get(metric, np.nan)
                imp[f"{metric}_gain_vs_static_equal"] = row.get(metric, np.nan) - static_row.get(metric, np.nan)
        improvement_rows.append(imp)

    improvement_df = pd.DataFrame(improvement_rows)
    save_dataframe(improvement_df, EXP3_RESULTS / "experiment3_improvement_vs_baselines.csv")

    # Best method by validation objective, then report its test confusion matrix.
    best_val_method = val_metrics_df.sort_values("objective_score", ascending=False).iloc[0]["method"]
    best_test_method = test_metrics_df.sort_values("objective_score", ascending=False).iloc[0]["method"]

    best_pred, best_prob = predictions_for_best[best_val_method]
    cm = confusion_matrix(y_test, best_pred)
    save_dataframe(pd.DataFrame(cm), EXP3_RESULTS / f"confusion_matrix_{best_val_method}.csv")
    plot_confusion(
        cm,
        EXP3_RESULTS / f"fig4_confusion_matrix_{best_val_method}.png",
        f"Confusion Matrix: {best_val_method}"
    )

    report = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
    save_json(report, EXP3_RESULTS / f"classification_report_{best_val_method}.json")
    pd.DataFrame(report).transpose().to_csv(
        EXP3_RESULTS / f"classification_report_{best_val_method}.csv",
        encoding="utf-8-sig",
    )

    # Plots.
    plot_metric_comparison(test_metrics_df, EXP3_RESULTS / "fig1_swarm_strategy_metric_comparison.png")
    plot_auc_comparison(test_metrics_df, EXP3_RESULTS / "fig2_swarm_strategy_auc_comparison.png")
    plot_weight_comparison(weights_df, team_names, EXP3_RESULTS / "fig3_cooperative_weight_policy_comparison.png")
    plot_convergence(all_trace, EXP3_RESULTS / "fig5_swarm_convergence_curves.png")

    # Stability summary.
    stability_rows = []
    for _, row in weights_df.iterrows():
        method = row["method"]
        w = np.array([row[t] for t in team_names], dtype=np.float64)
        stability_rows.append({
            "method": method,
            "weight_entropy": row["weight_entropy"],
            "max_weight": row["max_weight"],
            "min_weight": row["min_weight"],
            "weight_std": float(np.std(w)),
            "effective_number_of_teams": float(np.exp(row["weight_entropy"])),
            "dominant_team": team_names[int(np.argmax(w))],
        })
    stability_df = pd.DataFrame(stability_rows)
    save_dataframe(stability_df, EXP3_RESULTS / "experiment3_cooperation_stability_summary.csv")

    # Runtime metadata.
    runtime = {
        "experiment": "Experiment 3: Swarm Optimization and Cooperative Adaptation",
        "device": str(DEVICE),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "checkpoint": str(BEST_MODEL_FILE),
        "validation_file": str(VAL_FILE),
        "test_file": str(TEST_FILE),
        "output_folder": str(EXP3_RESULTS),
        "hybrid_selected_source": hybrid_source,
        "best_validation_method": best_val_method,
        "best_test_method": best_test_method,
    }
    save_json(runtime, EXP3_RESULTS / "experiment3_runtime_metadata.json")

    # Paper-ready summary.
    best_val_row = val_metrics_df[val_metrics_df["method"] == best_val_method].iloc[0]
    best_test_row = test_metrics_df[test_metrics_df["method"] == best_val_method].iloc[0]
    learned_test = learned_row

    auc_name = "roc_auc" if "roc_auc" in test_metrics_df.columns else "roc_auc_ovr"
    ap_name = "average_precision" if "average_precision" in test_metrics_df.columns else "average_precision_macro"

    f1_gain = best_test_row.get("f1_weighted", np.nan) - learned_test.get("f1_weighted", np.nan)
    auc_gain = best_test_row.get(auc_name, np.nan) - learned_test.get(auc_name, np.nan)
    bal_gain = best_test_row.get("balanced_accuracy", np.nan) - learned_test.get("balanced_accuracy", np.nan)

    dominant_policy = weights_df[weights_df["method"] == best_val_method].iloc[0]
    dominant_team = stability_df[stability_df["method"] == best_val_method].iloc[0]["dominant_team"]

    md = f"""# Experiment 3: Swarm Optimization and Cooperative Adaptation

## Objective
Experiment 3 evaluates whether swarm-guided cooperative adaptation improves the organizational coordination layer of the proposed Nash-cooperative multi-team framework. The experiment reuses the exact trained specialist-team architecture from Experiment 1 and optimizes only the cooperative team-weight policy using validation data. The resulting policies are then evaluated on the held-out internal test set.

## Compared Cooperation Strategies
The experiment compared:

- Static equal cooperation
- Learned Nash-cooperative weights from Experiment 1
- PSO-guided adaptive cooperation
- ACO-guided adaptive cooperation
- BCO-guided adaptive cooperation
- HHO-guided adaptive cooperation
- Hybrid swarm cooperation

## Best Validation-Guided Strategy
The best validation-guided strategy was **{best_val_method}**. On the internal test set, this strategy achieved:

- Accuracy: {best_test_row.get('accuracy', float('nan')):.4f}
- Balanced accuracy: {best_test_row.get('balanced_accuracy', float('nan')):.4f}
- Macro F1: {best_test_row.get('f1_macro', float('nan')):.4f}
- Weighted F1: {best_test_row.get('f1_weighted', float('nan')):.4f}
- {auc_name}: {best_test_row.get(auc_name, float('nan')):.4f}
- {ap_name}: {best_test_row.get(ap_name, float('nan')):.4f}
- Objective score: {best_test_row.get('objective_score', float('nan')):.4f}

Relative to the learned Nash-cooperative policy from Experiment 1, the selected swarm-adapted policy changed performance by:

- Weighted F1 gain: {f1_gain:.4f}
- {auc_name} gain: {auc_gain:.4f}
- Balanced accuracy gain: {bal_gain:.4f}

## Cooperative Weight Adaptation
The dominant team under the best validation-guided strategy was **{dominant_team}**. The effective number of active teams was {stability_df[stability_df['method'] == best_val_method].iloc[0]['effective_number_of_teams']:.4f}, indicating the extent to which the selected swarm policy concentrated or distributed organizational influence.

## Hybrid Swarm Source
The hybrid swarm policy selected **{hybrid_source}** as the strongest component optimizer under the validation objective.

## Interpretation
Experiment 3 directly tests the cooperative adaptation component of the Methods section. It shows whether population-based swarm search can improve or stabilize the organizational team-weight policy after specialist-team learning. If swarm-adapted policies improve weighted F1, AUC, or balanced accuracy, this supports the claim that swarm optimization contributes adaptive coordination beyond static cooperation. If performance remains similar, this still supports stability by showing that the learned Nash-cooperative policy was already near a strong internal coordination state.

## Generated Outputs
- experiment3_validation_metrics.csv
- experiment3_test_metrics.csv
- experiment3_cooperative_weight_policies.csv
- experiment3_swarm_convergence_traces.csv
- experiment3_improvement_vs_baselines.csv
- experiment3_cooperation_stability_summary.csv
- fig1_swarm_strategy_metric_comparison.png
- fig2_swarm_strategy_auc_comparison.png
- fig3_cooperative_weight_policy_comparison.png
- fig4_confusion_matrix_{best_val_method}.png
- fig5_swarm_convergence_curves.png
"""

    with open(EXP3_RESULTS / "experiment3_paper_ready_summary.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("\nExperiment 3 completed successfully.")
    print(f"Results folder: {EXP3_RESULTS}")
    print(f"Best validation-guided method: {best_val_method}")
    print(f"Best test objective method: {best_test_method}")
    print(f"Hybrid selected source: {hybrid_source}")
    print("\nTest metrics:")
    print(test_metrics_df[["method", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", auc_name, ap_name, "objective_score"]].to_string(index=False))
    print("\nMost important outputs:")
    print(f"  {EXP3_RESULTS / 'experiment3_test_metrics.csv'}")
    print(f"  {EXP3_RESULTS / 'experiment3_cooperative_weight_policies.csv'}")
    print(f"  {EXP3_RESULTS / 'experiment3_paper_ready_summary.md'}")


if __name__ == "__main__":
    main()

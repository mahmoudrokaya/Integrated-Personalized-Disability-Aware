# -*- coding: utf-8 -*-
"""
Experiment 2: Specialist-Team Contribution and Organizational Explainability
Paper: Nash-Cooperative Swarm Intelligence for Lightweight Edge Learning
       Through Multi-Team Adaptive Optimization

Purpose
-------
This script continues after Experiment 1 and evaluates whether the proposed
organizational model behaves as a transparent multi-team system rather than a
black-box monolithic classifier.

It generates report-ready evidence for:
1. Specialist-team contribution analysis
2. Team ablation and leave-one-team-out evaluation
3. Single-team-only evaluation
4. Cooperative-weight and utility interpretation
5. Permutation-based domain importance
6. Fairness-oriented subgroup performance by demographic variables
7. Prediction confidence and error analysis
8. Publication-ready figures and tables

Expected existing inputs
------------------------
Modeling-ready folder:
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/modeling_ready

Experiment 1 model:
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/Experiment1/Results/experiment1_best_model.pt

Outputs
-------
D:/47/472/New-Papers/Integrated Personalized Disability-Aware/Experiments/Experiment2/Results
"""

import os
import json
import time
import math
import random
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
)
from sklearn.inspection import permutation_importance
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
# 1. Configuration aligned with Methods Section 3
# ---------------------------------------------------------------------
CONFIG = {
    "seed": SEED,
    "batch_size": 128,
    "hidden_dim": 32,
    "dropout": 0.10,
    "num_learners_per_team": 3,
    "team_names": [
        "nutritional_metabolic",
        "physiological",
        "behavioral",
        "demographic_fairness",
        "functional_disability",
    ],
    "target_candidates": [
        "disability_stage",
        "target",
        "label",
        "y",
        "functional_risk",
        "merge01_vs_23",
    ],
    "domain_keywords": {
        "nutritional_metabolic": [
            "bmi", "hba1c", "glucose", "cholesterol", "hdl", "ldl",
            "triglycer", "weight", "height", "metabolic"
        ],
        "physiological": [
            "sbp", "dbp", "blood", "pressure", "grip", "strength",
            "pulse", "physio"
        ],
        "behavioral": [
            "smoking", "smoke", "activity", "met", "sedentary",
            "physical", "guideline", "behavior"
        ],
        "demographic_fairness": [
            "age", "sex", "gender", "race", "ethnicity", "education",
            "income", "poverty", "demographic"
        ],
        "functional_disability": [
            "difficulty", "disability", "adl", "mobility", "functional",
            "limitation", "stage", "phq", "arthritis", "polypharmacy",
            "medication"
        ],
    },
    "fallback_domain_order": [
        "demographic_fairness",
        "nutritional_metabolic",
        "physiological",
        "behavioral",
        "functional_disability",
    ],
}


# ---------------------------------------------------------------------
# 2. Utility functions
# ---------------------------------------------------------------------
def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_dataframe(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def find_target_column(df: pd.DataFrame) -> str:
    for col in CONFIG["target_candidates"]:
        if col in df.columns:
            return col
    # fallback: last column
    return df.columns[-1]


def ensure_numeric_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col].astype(str))[0]
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        if col != target_col:
            out[col] = out[col].fillna(out[col].median() if pd.api.types.is_numeric_dtype(out[col]) else 0)
    return out


def infer_feature_columns(train_df: pd.DataFrame, target_col: str) -> List[str]:
    exclude = {target_col, "SEQN", "seqn", "participant_id", "id"}
    features = [c for c in train_df.columns if c not in exclude]
    return features


def build_domain_map(feature_cols: List[str]) -> Dict[str, List[str]]:
    domain_map = {team: [] for team in CONFIG["team_names"]}
    lower_lookup = {c: c.lower() for c in feature_cols}

    assigned = set()
    for team, keywords in CONFIG["domain_keywords"].items():
        for col in feature_cols:
            low = lower_lookup[col]
            if col in assigned:
                continue
            if any(k in low for k in keywords):
                domain_map[team].append(col)
                assigned.add(col)

    # Assign still-unassigned features in a round-robin way to avoid empty teams.
    fallback = CONFIG["fallback_domain_order"]
    unassigned = [c for c in feature_cols if c not in assigned]
    for i, col in enumerate(unassigned):
        domain_map[fallback[i % len(fallback)]].append(col)

    # Guarantee no empty team by moving one feature from the largest domain.
    for team in CONFIG["team_names"]:
        if len(domain_map[team]) == 0:
            donor = max(domain_map, key=lambda k: len(domain_map[k]))
            moved = domain_map[donor].pop()
            domain_map[team].append(moved)

    return domain_map


def prepare_data():
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    target_col = find_target_column(train_df)

    train_df = ensure_numeric_frame(train_df, target_col)
    val_df = ensure_numeric_frame(val_df, target_col)
    test_df = ensure_numeric_frame(test_df, target_col)

    feature_cols = infer_feature_columns(train_df, target_col)
    domain_map = build_domain_map(feature_cols)

    # Class labels from train set, mapped consistently.
    classes = sorted(train_df[target_col].dropna().unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for df in [train_df, val_df, test_df]:
        df[target_col] = df[target_col].map(class_to_idx).astype(int)

    # Standardize using train statistics only.
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std().replace(0, 1.0)

    for df in [train_df, val_df, test_df]:
        df[feature_cols] = (df[feature_cols] - means) / stds
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    metadata = {
        "target_column": target_col,
        "feature_columns": feature_cols,
        "class_values_original": classes,
        "class_to_index": {str(k): int(v) for k, v in class_to_idx.items()},
        "domain_map": domain_map,
        "num_classes": len(classes),
    }
    save_json(metadata, EXP2_RESULTS / "experiment2_data_metadata.json")

    return train_df, val_df, test_df, metadata


class MultidomainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_col: str, domain_map: Dict[str, List[str]]):
        self.y = torch.tensor(df[target_col].values, dtype=torch.long)
        self.domain_data = {}
        for team, cols in domain_map.items():
            self.domain_data[team] = torch.tensor(df[cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = {team: data[idx] for team, data in self.domain_data.items()}
        return x, self.y[idx]


# ---------------------------------------------------------------------
# 3. Model architecture matching Experiment 1
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
        self.learner_logits = nn.Parameter(torch.zeros(n_learners))

    def forward(self, x):
        learner_outputs = torch.stack([learner(x) for learner in self.learners], dim=1)
        learner_weights = torch.softmax(self.learner_logits, dim=0)
        team_output = torch.sum(learner_outputs * learner_weights.view(1, -1, 1), dim=1)
        return team_output, learner_outputs, learner_weights


class NashCooperativeOrganizationalModel(nn.Module):
    def __init__(self, domain_map: Dict[str, List[str]], hidden_dim: int, num_classes: int, n_learners: int, dropout: float):
        super().__init__()
        self.team_names = CONFIG["team_names"]
        self.domain_map = domain_map
        self.teams = nn.ModuleDict({
            team: SpecialistTeam(len(domain_map[team]), hidden_dim, num_classes, n_learners, dropout)
            for team in self.team_names
        })
        self.team_logits = nn.Parameter(torch.zeros(len(self.team_names)))

    def forward(self, x_dict, mask_team: Optional[str] = None, only_team: Optional[str] = None):
        team_outputs = []
        learner_weight_dict = {}
        raw_team_dict = {}

        for team in self.team_names:
            out, learner_outputs, learner_weights = self.teams[team](x_dict[team])
            team_outputs.append(out)
            learner_weight_dict[team] = learner_weights
            raw_team_dict[team] = out

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
# 4. Evaluation functions
# ---------------------------------------------------------------------
@torch.no_grad()
def predict_model(model, loader, mask_team=None, only_team=None):
    model.eval()
    all_y, all_pred, all_prob, all_logits = [], [], [], []
    all_team_probs = {team: [] for team in CONFIG["team_names"]}

    total_loss = 0.0
    n = 0

    for x_dict, y in loader:
        x_dict = {k: v.to(DEVICE) for k, v in x_dict.items()}
        y = y.to(DEVICE)

        logits, stacked, weights, raw_team_dict, _ = model(
            x_dict, mask_team=mask_team, only_team=only_team
        )

        loss = F.cross_entropy(logits, y)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        all_y.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob.append(prob.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

        for team, team_logits in raw_team_dict.items():
            all_team_probs[team].append(torch.softmax(team_logits, dim=1).cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    logits = np.concatenate(all_logits)
    team_probs = {team: np.concatenate(parts) for team, parts in all_team_probs.items()}

    return {
        "loss": total_loss / max(n, 1),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "logits": logits,
        "team_probs": team_probs,
    }


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    num_classes = y_prob.shape[1]
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
        if num_classes == 2:
            out["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            out["average_precision"] = average_precision_score(y_true, y_prob[:, 1])
        else:
            out["roc_auc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            out["average_precision_macro"] = average_precision_score(y_bin, y_prob, average="macro")
    except Exception:
        pass

    return out


def evaluate_named_setting(model, loader, setting_name: str, mask_team=None, only_team=None):
    result = predict_model(model, loader, mask_team=mask_team, only_team=only_team)
    metrics = compute_metrics(result["y_true"], result["y_pred"], result["y_prob"])
    metrics["loss"] = result["loss"]
    metrics["setting"] = setting_name
    return result, metrics


def team_prediction_agreement(base_pred, team_prob_dict):
    rows = []
    for team, prob in team_prob_dict.items():
        team_pred = prob.argmax(axis=1)
        rows.append({
            "team": team,
            "agreement_with_global_prediction": accuracy_score(base_pred, team_pred),
            "mean_team_confidence": float(np.max(prob, axis=1).mean()),
        })
    return pd.DataFrame(rows)


def compute_team_error_overlap(y_true, base_pred, team_prob_dict):
    rows = []
    global_error = base_pred != y_true
    for team, prob in team_prob_dict.items():
        team_pred = prob.argmax(axis=1)
        team_error = team_pred != y_true
        both_error = np.logical_and(global_error, team_error).mean()
        team_unique_error = np.logical_and(~global_error, team_error).mean()
        global_unique_error = np.logical_and(global_error, ~team_error).mean()
        rows.append({
            "team": team,
            "team_error_rate": float(team_error.mean()),
            "global_error_rate": float(global_error.mean()),
            "shared_error_rate": float(both_error),
            "team_only_error_rate": float(team_unique_error),
            "global_only_error_rate": float(global_unique_error),
        })
    return pd.DataFrame(rows)


def fairness_subgroup_metrics(df: pd.DataFrame, y_true, y_pred, y_prob, candidate_cols: List[str]):
    rows = []
    for col in candidate_cols:
        if col not in df.columns:
            continue
        values = df[col].copy()

        # Use quantile groups for continuous/high-cardinality variables.
        if pd.api.types.is_numeric_dtype(values) and values.nunique() > 6:
            try:
                groups = pd.qcut(values, q=4, duplicates="drop").astype(str)
            except Exception:
                groups = values.astype(str)
        else:
            groups = values.astype(str)

        for group in sorted(groups.unique()):
            idx = groups == group
            if idx.sum() < 20:
                continue
            yt = y_true[idx.values]
            yp = y_pred[idx.values]
            pr = y_prob[idx.values]
            met = compute_metrics(yt, yp, pr)
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
# 5. Plotting functions
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
    plot_df = df[[label_col] + metric_cols].copy()
    x = np.arange(len(plot_df))
    width = 0.8 / len(metric_cols)

    plt.figure(figsize=(11, 6))
    for i, metric in enumerate(metric_cols):
        plt.bar(x + i * width, plot_df[metric], width=width, label=metric)

    plt.xticks(x + width * (len(metric_cols) - 1) / 2, plot_df[label_col], rotation=30, ha="right")
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

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_team_probability_correlation(team_probs, path):
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
    plt.title("Correlation Between Specialist-Team Prediction Signals")
    for i in range(len(team_names)):
        for j in range(len(team_names)):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")
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


# ---------------------------------------------------------------------
# 6. Main experiment
# ---------------------------------------------------------------------
def main():
    print("Starting Experiment 2: Specialist-Team Contribution and Organizational Explainability")
    print(f"Device: {DEVICE}")
    print(f"Output folder: {EXP2_RESULTS}")

    if not BEST_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Experiment 1 model not found: {BEST_MODEL_FILE}\n"
            "Run Experiment 1 first or check the file path."
        )

    train_df, val_df, test_df, metadata = prepare_data()
    target_col = metadata["target_column"]
    domain_map = metadata["domain_map"]
    num_classes = metadata["num_classes"]

    test_dataset = MultidomainDataset(test_df, target_col, domain_map)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    model = NashCooperativeOrganizationalModel(
        domain_map=domain_map,
        hidden_dim=CONFIG["hidden_dim"],
        num_classes=num_classes,
        n_learners=CONFIG["num_learners_per_team"],
        dropout=CONFIG["dropout"],
    ).to(DEVICE)

    checkpoint = torch.load(BEST_MODEL_FILE, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise ValueError("Unsupported checkpoint format.")

    model.eval()

    # Baseline full cooperative model.
    base_result, base_metrics = evaluate_named_setting(model, test_loader, "full_cooperative_model")

    # Save classification report and confusion matrix.
    report = classification_report(
        base_result["y_true"],
        base_result["y_pred"],
        output_dict=True,
        zero_division=0,
    )
    save_json(report, EXP2_RESULTS / "classification_report_full_model.json")
    pd.DataFrame(report).transpose().to_csv(
        EXP2_RESULTS / "classification_report_full_model.csv",
        encoding="utf-8-sig",
    )

    cm = confusion_matrix(base_result["y_true"], base_result["y_pred"])
    pd.DataFrame(cm).to_csv(EXP2_RESULTS / "confusion_matrix_full_model.csv", index=False)
    plot_confusion(cm, EXP2_RESULTS / "fig1_confusion_matrix_full_model.png")

    # Final team weights.
    with torch.no_grad():
        final_team_weights = torch.softmax(model.team_logits, dim=0).cpu().numpy()
    team_weights_df = pd.DataFrame({
        "team": CONFIG["team_names"],
        "cooperative_weight": final_team_weights,
        "num_features": [len(domain_map[t]) for t in CONFIG["team_names"]],
        "features": ["; ".join(domain_map[t]) for t in CONFIG["team_names"]],
    })
    save_dataframe(team_weights_df, EXP2_RESULTS / "team_cooperative_weights.csv")
    plot_bar(
        team_weights_df,
        "team",
        "cooperative_weight",
        "Final Nash-Cooperative Specialist-Team Weights",
        "Cooperative weight",
        EXP2_RESULTS / "fig2_final_team_weights.png",
    )

    # Single-team-only evaluation.
    single_rows = []
    for team in CONFIG["team_names"]:
        _, met = evaluate_named_setting(model, test_loader, f"only_{team}", only_team=team)
        met["team"] = team
        single_rows.append(met)
    single_df = pd.DataFrame(single_rows)
    save_dataframe(single_df, EXP2_RESULTS / "single_team_only_metrics.csv")

    plot_grouped_metrics(
        single_df,
        ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"],
        "team",
        "Single-Team-Only Performance",
        EXP2_RESULTS / "fig3_single_team_only_performance.png",
    )

    # Leave-one-team-out ablation.
    ablation_rows = []
    for team in CONFIG["team_names"]:
        _, met = evaluate_named_setting(model, test_loader, f"without_{team}", mask_team=team)
        met["removed_team"] = team
        ablation_rows.append(met)
    ablation_df = pd.DataFrame(ablation_rows)

    # Add performance drops relative to full model.
    for metric in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "average_precision"]:
        if metric in ablation_df.columns and metric in base_metrics:
            ablation_df[f"{metric}_drop_vs_full"] = base_metrics[metric] - ablation_df[metric]

    save_dataframe(ablation_df, EXP2_RESULTS / "leave_one_team_out_ablation_metrics.csv")

    drop_cols = [c for c in ["accuracy_drop_vs_full", "f1_weighted_drop_vs_full", "roc_auc_drop_vs_full"] if c in ablation_df.columns]
    if drop_cols:
        plot_grouped_metrics(
            ablation_df,
            drop_cols,
            "removed_team",
            "Performance Drop After Removing Each Specialist Team",
            EXP2_RESULTS / "fig4_leave_one_team_out_drops.png",
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
        EXP2_RESULTS / "fig5_team_global_agreement.png",
    )

    error_df = compute_team_error_overlap(
        base_result["y_true"],
        base_result["y_pred"],
        base_result["team_probs"],
    )
    save_dataframe(error_df, EXP2_RESULTS / "team_error_overlap.csv")
    plot_grouped_metrics(
        error_df,
        ["team_error_rate", "shared_error_rate", "team_only_error_rate"],
        "team",
        "Specialist-Team Error Overlap Analysis",
        EXP2_RESULTS / "fig6_team_error_overlap.png",
    )

    # Team signal correlation and confidence distribution.
    plot_team_probability_correlation(
        base_result["team_probs"],
        EXP2_RESULTS / "fig7_team_prediction_signal_correlation.png",
    )
    plot_confidence_distribution(
        base_result["y_true"],
        base_result["y_pred"],
        base_result["y_prob"],
        EXP2_RESULTS / "fig8_prediction_confidence_distribution.png",
    )

    # Domain-level permutation analysis.
    # This is done by shuffling all features of one domain at a time.
    permutation_rows = []
    base_auc = base_metrics.get("roc_auc", np.nan)
    base_f1 = base_metrics.get("f1_weighted", np.nan)
    base_acc = base_metrics.get("accuracy", np.nan)

    for team in CONFIG["team_names"]:
        shuffled_df = test_df.copy()
        for col in domain_map[team]:
            shuffled_df[col] = np.random.permutation(shuffled_df[col].values)

        shuffled_dataset = MultidomainDataset(shuffled_df, target_col, domain_map)
        shuffled_loader = DataLoader(shuffled_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
        _, met = evaluate_named_setting(model, shuffled_loader, f"permuted_{team}")

        permutation_rows.append({
            "permuted_domain": team,
            "accuracy": met.get("accuracy", np.nan),
            "f1_weighted": met.get("f1_weighted", np.nan),
            "roc_auc": met.get("roc_auc", np.nan),
            "accuracy_drop": base_acc - met.get("accuracy", np.nan),
            "f1_weighted_drop": base_f1 - met.get("f1_weighted", np.nan),
            "roc_auc_drop": base_auc - met.get("roc_auc", np.nan) if not np.isnan(base_auc) else np.nan,
        })

    permutation_df = pd.DataFrame(permutation_rows)
    save_dataframe(permutation_df, EXP2_RESULTS / "domain_permutation_importance.csv")
    plot_grouped_metrics(
        permutation_df,
        ["accuracy_drop", "f1_weighted_drop", "roc_auc_drop"],
        "permuted_domain",
        "Domain-Level Permutation Importance",
        EXP2_RESULTS / "fig9_domain_permutation_importance.png",
    )

    # Fairness/domain subgroup performance.
    fairness_candidates = [
        c for c in metadata["feature_columns"]
        if any(k in c.lower() for k in ["sex", "gender", "race", "ethnicity", "education", "income", "poverty", "age"])
    ]
    fairness_df = fairness_subgroup_metrics(
        test_df,
        base_result["y_true"],
        base_result["y_pred"],
        base_result["y_prob"],
        fairness_candidates,
    )
    if not fairness_df.empty:
        save_dataframe(fairness_df, EXP2_RESULTS / "demographic_subgroup_performance.csv")

        # Summarize disparities by attribute.
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
            EXP2_RESULTS / "fig10_demographic_f1_gap.png",
        )

    # Save full-model metrics.
    base_metrics_out = dict(base_metrics)
    base_metrics_out["setting"] = "full_cooperative_model"
    save_json(base_metrics_out, EXP2_RESULTS / "full_model_metrics.json")
    pd.DataFrame([base_metrics_out]).to_csv(
        EXP2_RESULTS / "full_model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Integrated summary table.
    summary_rows = []
    for _, r in team_weights_df.iterrows():
        team = r["team"]
        single = single_df[single_df["team"] == team].iloc[0].to_dict()
        ablated = ablation_df[ablation_df["removed_team"] == team].iloc[0].to_dict()
        permuted = permutation_df[permutation_df["permuted_domain"] == team].iloc[0].to_dict()
        agreement = agreement_df[agreement_df["team"] == team].iloc[0].to_dict()

        summary_rows.append({
            "team": team,
            "cooperative_weight": r["cooperative_weight"],
            "num_features": r["num_features"],
            "single_team_accuracy": single.get("accuracy", np.nan),
            "single_team_f1_weighted": single.get("f1_weighted", np.nan),
            "single_team_auc": single.get("roc_auc", np.nan),
            "drop_when_removed_accuracy": ablated.get("accuracy_drop_vs_full", np.nan),
            "drop_when_removed_f1_weighted": ablated.get("f1_weighted_drop_vs_full", np.nan),
            "drop_when_removed_auc": ablated.get("roc_auc_drop_vs_full", np.nan),
            "permutation_auc_drop": permuted.get("roc_auc_drop", np.nan),
            "agreement_with_global_prediction": agreement.get("agreement_with_global_prediction", np.nan),
        })

    integrated_df = pd.DataFrame(summary_rows)
    save_dataframe(integrated_df, EXP2_RESULTS / "experiment2_integrated_team_explainability_summary.csv")

    # System/runtime metadata.
    runtime_metadata = {
        "experiment": "Experiment 2: Specialist-Team Contribution and Organizational Explainability",
        "device": str(DEVICE),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "input_model": str(BEST_MODEL_FILE),
        "test_file": str(TEST_FILE),
        "output_folder": str(EXP2_RESULTS),
    }
    save_json(runtime_metadata, EXP2_RESULTS / "experiment2_runtime_metadata.json")

    # Paper-ready markdown report.
    strongest_weight_team = team_weights_df.sort_values("cooperative_weight", ascending=False).iloc[0]
    strongest_single_team = single_df.sort_values("f1_weighted", ascending=False).iloc[0]
    strongest_ablation = integrated_df.sort_values("drop_when_removed_auc", ascending=False).iloc[0] if "drop_when_removed_auc" in integrated_df.columns else integrated_df.iloc[0]
    strongest_perm = permutation_df.sort_values("roc_auc_drop", ascending=False).iloc[0]

    report_md = f"""# Experiment 2: Specialist-Team Contribution and Organizational Explainability

## Objective
Experiment 2 evaluates whether the proposed Nash-cooperative organizational framework behaves as an interpretable multi-team intelligence system rather than a monolithic classifier. The experiment analyzes specialist-team contribution, domain-level importance, leave-one-team-out robustness, single-team predictive capacity, team agreement, confidence behavior, and demographic subgroup consistency.

## Full Cooperative Model
The full cooperative model achieved the following internal test performance:

- Accuracy: {base_metrics.get('accuracy', float('nan')):.4f}
- Balanced accuracy: {base_metrics.get('balanced_accuracy', float('nan')):.4f}
- Macro F1: {base_metrics.get('f1_macro', float('nan')):.4f}
- Weighted F1: {base_metrics.get('f1_weighted', float('nan')):.4f}
- ROC-AUC: {base_metrics.get('roc_auc', float('nan')):.4f}
- Average precision: {base_metrics.get('average_precision', float('nan')):.4f}
- Mean prediction confidence: {base_metrics.get('mean_prediction_confidence', float('nan')):.4f}

## Specialist-Team Cooperative Weights
The highest final Nash-cooperative weight was assigned to the **{strongest_weight_team['team']}** team with a weight of {strongest_weight_team['cooperative_weight']:.4f}. This indicates that the adaptive organizational controller assigned the greatest contribution to this specialist domain during internal disability-health prediction.

## Single-Team Evaluation
The strongest single-team model was **{strongest_single_team['team']}**, with weighted F1 = {strongest_single_team.get('f1_weighted', float('nan')):.4f} and ROC-AUC = {strongest_single_team.get('roc_auc', float('nan')):.4f}. This analysis helps identify whether each team carries useful independent predictive information.

## Leave-One-Team-Out Ablation
The largest ROC-AUC degradation after team removal was observed when removing **{strongest_ablation['team']}**, with an AUC drop of {strongest_ablation.get('drop_when_removed_auc', float('nan')):.4f}. This provides direct evidence of the organizational contribution of that specialist team to the cooperative prediction.

## Domain Permutation Importance
The largest domain-level permutation effect occurred for **{strongest_perm['permuted_domain']}**, with an AUC drop of {strongest_perm.get('roc_auc_drop', float('nan')):.4f}. This result supports domain-level explainability by measuring the sensitivity of the organizational prediction to each specialist domain.

## Generated Outputs
The following report-ready tables and figures were generated:

- full_model_metrics.csv
- classification_report_full_model.csv
- confusion_matrix_full_model.csv
- team_cooperative_weights.csv
- single_team_only_metrics.csv
- leave_one_team_out_ablation_metrics.csv
- team_prediction_agreement.csv
- team_error_overlap.csv
- domain_permutation_importance.csv
- demographic_subgroup_performance.csv
- demographic_fairness_gaps.csv
- experiment2_integrated_team_explainability_summary.csv
- fig1_confusion_matrix_full_model.png
- fig2_final_team_weights.png
- fig3_single_team_only_performance.png
- fig4_leave_one_team_out_drops.png
- fig5_team_global_agreement.png
- fig6_team_error_overlap.png
- fig7_team_prediction_signal_correlation.png
- fig8_prediction_confidence_distribution.png
- fig9_domain_permutation_importance.png
- fig10_demographic_f1_gap.png

## Interpretation for Manuscript
Experiment 2 provides explainability evidence at the organizational level. It shows which specialist teams contribute most strongly, whether teams retain independent predictive value, how performance changes when individual teams are removed, and whether demographic subgroup performance remains consistent. These outputs directly support the Methods section, where the proposed framework is defined as a cooperative multi-team architecture with interpretable specialist-team influence and Nash-adaptive contribution weights.
"""

    with open(EXP2_RESULTS / "experiment2_paper_ready_summary.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\nExperiment 2 completed successfully.")
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

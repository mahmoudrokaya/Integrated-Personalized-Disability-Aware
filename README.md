# Integrated Personalized Disability-Aware Health Intelligence

## Project Overview

This repository contains the complete codebase, datasets, and experiment outputs for the paper:

# **Integrated Personalized Disability-Aware Health Intelligence**

The project presents a data-driven intelligent framework for disability-aware health prediction using **NHANES** data through a multi-stage computational pipeline combining:

- multidomain health-data integration
- structured preprocessing and harmonization
- specialist-team cooperative learning
- explainability analysis
- adaptive optimization
- lightweight computational efficiency analysis
- temporal external validation

The repository was organized to support **full computational reproducibility** of all experiments reported in the manuscript.

---

# Repository Structure

This repository is organized into multiple branches for clarity and reproducibility.

---

# Main Branch

The **main branch** contains:

- all Python source code
- preprocessing scripts
- modeling scripts
- experiment scripts
- utilities
- manuscript-related implementation files
- documentation

This branch represents the complete computational workflow of the study.

---

# Data Branch

## Branch name:
`data`

The **data branch** contains all datasets used in this work.

This includes:

### Raw data
- original NHANES source files
- downloaded later-cycle NHANES files

### Processed data
- harmonized cycle-level processed files
- corrected pooled datasets
- filled datasets
- validation outputs

### Modeling-ready data
- `train_2011_2018.csv`
- `val_2011_2018.csv`
- `test_2011_2018.csv`
- `external_2017_2023.csv`

The data branch is provided to support complete reproducibility of all experiments.

---

# Experiment Result Branches

Each experiment has a dedicated branch containing all generated outputs.

---

# `experiment1`

Contains:

- trained model checkpoints
- training logs
- learning curves
- confusion matrices
- evaluation metrics
- CSV summaries
- publication-ready figures

### Corresponds to:
**Experiment 1 — Organizational Training and Internal Cooperative Learning**

---

# `experiment2`

Contains:

- explainability outputs
- specialist-team contribution analysis
- team influence summaries
- permutation importance outputs
- ablation-related results
- generated figures
- report tables

### Corresponds to:
**Experiment 2 — Specialist-Team Explainability and Cooperative Contribution Analysis**

---

# `experiment3`

Contains:

- swarm optimization outputs
- cooperative adaptation metrics
- convergence traces
- cooperative-weight policies
- comparison summaries
- plots and report tables

### Corresponds to:
**Experiment 3 — Swarm Optimization and Cooperative Adaptation**

---

# `experiment4`

Contains:

- efficiency metrics
- latency analysis
- throughput evaluation
- model size analysis
- memory footprint reports
- lightweight deployment benchmarks
- publication figures

### Corresponds to:
**Experiment 4 — Lightweight Edge Efficiency**

---

# `experiment5`

Contains:

- temporal validation outputs
- external later-cycle evaluation
- temporal shift analysis
- subgroup analysis
- feature availability reports
- performance comparison tables
- generated figures

### Corresponds to:
**Experiment 5 — Temporal Organizational Generalization**

---

# Scientific Workflow

The full study follows the methodological order described in the manuscript.

---

## Stage 1 — Data Acquisition

Scripts:
- `nhanes_2019_2026_explore_download.py`

Purpose:
- explore later NHANES releases
- identify required variables
- download raw files
- prepare temporal data integration

---

## Stage 2 — Missing Data Processing and Harmonization

Scripts:
- `nhanes_missing_core_download_and_process.py`
- `run_later_cycles_preprocessing_patched.py`

Purpose:
- missing-data handling
- harmonization across cycles
- feature consistency correction
- pooled later-cycle preparation

---

## Stage 3 — Later-Cycle Inspection and Validation

Scripts:
- `nhanes_later_cycles_inspect_process.py`
- `audit_later_cycles_dataset.py`

Purpose:
- dataset auditing
- structural validation
- feature overlap verification
- target compatibility checking

---

## Stage 4 — Modeling-Ready Dataset Generation

Scripts:
- `prepare_modeling_datasets.py`
- `prepare_modeling_datasets_v2.py`

Purpose:
- feature alignment
- domain mapping
- leakage-safe split generation
- train/validation/test preparation
- external temporal validation preparation

---

# Experiments

---

# Experiment 1 — Organizational Internal Learning

Scripts:
- `experiment1_organizational_training_reporting.py`
- `experiment1_organizational_training_reporting_v2.py`

Main objective:

Train the internal disability-aware learning architecture on NHANES 2011–2018 data.

Outputs include:

- trained models
- performance metrics
- learning curves
- specialist-team weights

---

# Experiment 2 — Explainability and Team Contribution

Scripts:
- `experiment2_specialist_team_explainability.py`
- `experiment2_specialist_team_explainability_v2.py`
- `experiment2_specialist_team_explainability_v3.py`
- `generate_experiment2_domain_permutation_importance.py`

Main objective:

Evaluate:

- specialist-team contribution
- explainability
- feature importance
- domain-level influence
- cooperative weight interpretation

---

# Experiment 3 — Cooperative Adaptation

Script:
- `experiment3_swarm_optimization_cooperative_adaptation.py`

Main objective:

Evaluate adaptive cooperative optimization using swarm-based coordination.

---

# Experiment 4 — Lightweight Efficiency Evaluation

Script:
- `experiment4_lightweight_edge_efficiency.py`

Main objective:

Evaluate:

- computational efficiency
- memory footprint
- latency
- throughput
- edge deployment suitability

---

# Experiment 5 — Temporal External Validation

Script:
- `experiment5_temporal_organizational_generalization.py`

Main objective:

Evaluate model robustness under:

- temporal distribution shift
- feature drift
- partial observability
- target harmonization differences

using later NHANES cycles.

---

# Reproducibility

This repository is released to support **full reproducibility** of the study.

The implementation includes:

- fixed random seeds
- deterministic training
- saved model checkpoints
- metadata persistence
- harmonized preprocessing
- explicit feature alignment
- reproducible train/validation/test splits
- external validation datasets
- experiment-specific saved outputs

---

# Requirements

Main dependencies include:

```bash
Python >= 3.10
numpy
pandas
scikit-learn
matplotlib
torch
openpyxl
pyreadstat
requests
beautifulsoup4
tqdm

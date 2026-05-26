# Experiment 2: Specialist-Team Contribution and Organizational Explainability

## Objective
Experiment 2 evaluates the interpretability of the trained Nash-cooperative organizational model from Experiment 1. Unlike a separate training experiment, this analysis directly reloads the exact Experiment 1 checkpoint, architecture metadata, specialist-team decomposition, feature ordering, and scaling references. This ensures that all explainability outputs are scientifically consistent with the trained organizational framework.

## Full Cooperative Model Performance
The full cooperative model achieved:

- Accuracy: 0.7371
- Balanced accuracy: 0.5242
- Macro F1: 0.4962
- Weighted F1: 0.6845
- ROC-AUC: nan
- Average precision: nan
- Mean prediction confidence: 0.5796

## Specialist-Team Contribution
The highest Nash-cooperative weight was assigned to the **demographic_fairness** team, with a final weight of 0.5582. This indicates that this specialist domain exerted the strongest learned organizational influence in the final cooperative prediction.

## Single-Team Explainability
The strongest single-team-only configuration was **physiological**, achieving weighted F1 = 0.6253. This measures the independent predictive capacity of each specialist population when isolated from the cooperative organization.

## Leave-One-Team-Out Contribution
The largest degradation after removing one team occurred for **physiological**, with weighted-F1 drop = 0.2988. This provides direct evidence of the contribution of that specialist team to the full organizational model.

## Domain Permutation Importance
The strongest domain-level permutation effect was observed for **physiological**, with weighted-F1 drop = 0.2988. This indicates that disturbing this domain caused the largest reduction in cooperative prediction quality.

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

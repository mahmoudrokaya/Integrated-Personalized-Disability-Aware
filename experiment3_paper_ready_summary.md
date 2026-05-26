# Experiment 3: Swarm Optimization and Cooperative Adaptation

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
The best validation-guided strategy was **PSO_Adaptive_Cooperation**. On the internal test set, this strategy achieved:

- Accuracy: 0.7276
- Balanced accuracy: 0.5295
- Macro F1: 0.5138
- Weighted F1: 0.6970
- roc_auc_ovr: 0.8880
- average_precision_macro: 0.5860
- Objective score: 0.7029

Relative to the learned Nash-cooperative policy from Experiment 1, the selected swarm-adapted policy changed performance by:

- Weighted F1 gain: 0.0125
- roc_auc_ovr gain: 0.0130
- Balanced accuracy gain: 0.0053

## Cooperative Weight Adaptation
The dominant team under the best validation-guided strategy was **physiological**. The effective number of active teams was 4.7632, indicating the extent to which the selected swarm policy concentrated or distributed organizational influence.

## Hybrid Swarm Source
The hybrid swarm policy selected **PSO** as the strongest component optimizer under the validation objective.

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
- fig4_confusion_matrix_PSO_Adaptive_Cooperation.png
- fig5_swarm_convergence_curves.png

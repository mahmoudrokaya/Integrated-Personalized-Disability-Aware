# Experiment 4: Lightweight Edge Efficiency

## Objective
Experiment 4 evaluates the lightweight edge-learning efficiency of the proposed Nash-cooperative organizational framework. The experiment reloads the exact specialist-team architecture from Experiment 1 and measures model size, memory footprint, approximate computational cost, inference latency, throughput, batch-scaling behavior, and performance-efficiency trade-offs across cooperation policies.

## Model Size and Computational Cost
The trained organizational model contains:

- Total parameters: 2,528
- Trainable parameters: 2,528
- FP32 parameter memory: 9.8750 KB
- Approximate multiply-add operations per sample: 2,308
- Specialist teams: 5
- Learners per team: 3
- Hidden dimension: 16
- Input features: 26

These values confirm that the model remains lightweight and suitable for edge-oriented inference.

## Latency and Throughput
The fastest cooperation policy was **Experiment1_Learned_Nash**, with latency of 0.009752 ms per sample and throughput of 102542.00 samples per second.

The strongest weighted-F1 efficiency policy was **ACO_Adaptive_Cooperation**, achieving weighted F1 per millisecond of 71.753716.

The strongest AUC-efficiency policy was **ACO_Adaptive_Cooperation**, achieving AUC per millisecond of 90.808349.

## Edge Deployment Profile
The lightweight profile showed:

- Parameters < 10,000: True
- Parameters < 100,000: True
- FP32 memory < 1 MB: True
- Approximate multiply-adds < 100,000 per sample: True
- Approximate multiply-adds < 1,000,000 per sample: True

## Performance-Efficiency Summary
The cooperation-policy efficiency table reports accuracy, balanced accuracy, macro F1, weighted F1, roc_auc_ovr, average_precision_macro, latency, throughput, parameter count, and efficiency-normalized scores. This allows the paper to report not only predictive performance but also deployability under lightweight edge constraints.

## Interpretation
Experiment 4 directly supports the lightweight edge-learning claim in the paper title and Methods section. The results show that the organizational framework maintains a small parameter footprint, low memory cost, low approximate computational complexity, and fast CPU inference while preserving the predictive and cooperative behavior demonstrated in Experiments 1–3. This strengthens the claim that Nash-cooperative multi-team intelligence can operate as an edge-suitable organizational learning system rather than a computationally heavy centralized model.

## Generated Outputs
- experiment4_model_size_summary.csv
- experiment4_team_parameter_summary.csv
- experiment4_approximate_madds_summary.csv
- experiment4_policy_efficiency_metrics.csv
- experiment4_batch_scaling_latency.csv
- experiment4_lightweight_deployment_profile.csv
- fig1_parameter_count_by_component.png
- fig2_specialist_team_parameter_distribution.png
- fig3_approximate_madds_by_component.png
- fig4_latency_across_cooperation_policies.png
- fig5_batch_size_latency_scaling.png
- fig5_batch_size_throughput_scaling.png
- fig6_weighted_f1_vs_latency.png
- fig7_auc_vs_latency.png
- fig8_weighted_f1_per_ms.png

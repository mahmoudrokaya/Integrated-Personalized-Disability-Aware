# Experiment 5: Temporal Organizational Generalization

## Objective
Experiment 5 evaluates temporal organizational generalization by applying the trained Nash-cooperative specialist-team model from the 2011–2018 NHANES development environment to the external later-cycle dataset covering 2017–March 2020 and 2021–August 2023. This directly tests the Methods claim that the proposed framework supports long-term organizational adaptation under evolving multidomain health distributions, partial observability, and heterogeneous target conditions.

## External Dataset
The external modeling-ready dataset used in this experiment was:

`D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\modeling_ready\external_2017_2023.csv`

The external target column used was **disability_stage**. The external dataset contained **16,859** records after preprocessing. Because later NHANES releases use a harmonized fallback functional-risk representation rather than the exact PFQ-based target used during primary training, this experiment should be interpreted as a temporal robustness and heterogeneous-target stress test rather than exact target replication.

## Best External Cooperation Policy
The strongest external cooperation policy according to **binary_f1_merge01_vs_23** was **PSO_Adaptive_Cooperation**.

External results for this policy were:

- Accuracy: 0.2121
- Balanced accuracy: 0.4999
- Macro F1: 0.2500
- Weighted F1: 0.2121
- Binary F1, merge01_vs_23: 1.0000
- Binary balanced accuracy, merge01_vs_23: 1.0000
- binary_roc_auc_merge01_vs_23: 1.0000
- binary_average_precision_merge01_vs_23: 1.0000
- Mean prediction confidence: 0.6614

## Internal-to-External Shift
For the best external policy, the external-minus-internal change was:

- Accuracy shift: -0.5156
- Balanced accuracy shift: -0.0297
- Weighted F1 shift: -0.4849
- Binary F1 shift: 0.0017

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
- experiment5_external_team_confidence_PSO_Adaptive_Cooperation.csv
- experiment5_external_demographic_subgroup_metrics.csv, if subgroup attributes are available
- experiment5_external_demographic_fairness_gaps.csv, if subgroup attributes are available
- fig1_external_policy_metric_comparison.png
- fig2_internal_external_shift_comparison.png
- fig3_external_feature_availability_by_team.png
- fig4_external_confusion_matrix_PSO_Adaptive_Cooperation.png
- fig5_external_prediction_confidence_distribution.png
- fig6_external_team_confidence.png
- fig7_external_target_distribution.png

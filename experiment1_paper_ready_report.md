# Experiment 1: Organizational Training and Internal Cooperative Learning

## Aim
This experiment evaluates the internal learning behavior of the proposed Nash-cooperative organizational framework using the modeling-ready NHANES 2011-2018 training, validation, and test splits. The experiment directly reflects the Methods architecture by decomposing multidomain health records into specialist teams, training lightweight learners inside each team, and integrating team predictions through adaptive Nash-cooperative weighting.

## Data
- Training records: 7349
- Validation records: 1575
- Test records: 1575
- Number of features: 26
- Target column: disability_stage
- Number of classes: 4

## Main validation results
- Accuracy: 0.7644
- Balanced accuracy: 0.5260
- Macro-F1: 0.5030
- Weighted-F1: 0.7138
- ROC-AUC: 0.8866
- Average precision: 0.5823

## Main internal test results
- Accuracy: 0.7390
- Balanced accuracy: 0.5213
- Macro-F1: 0.4885
- Weighted-F1: 0.6803
- ROC-AUC: 0.8905
- Average precision: 0.5903
- Matthews correlation coefficient: 0.6228
- Mean prediction confidence: 0.7366

## Specialist-team contribution analysis
- nutritional_metabolic: cooperative weight = 0.1071, mean utility = -1.2636
- physiological: cooperative weight = 0.5582, mean utility = -0.5321
- behavioral: cooperative weight = 0.1053, mean utility = -1.4342
- demographic_fairness: cooperative weight = 0.1178, mean utility = -1.3078
- functional_disability: cooperative weight = 0.1116, mean utility = -1.5282

## Lightweight edge-learning profile
- Total parameters: 2528
- Trainable parameters: 2528
- Mean inference latency per sample: 0.00000606 seconds
- Median inference latency per sample: 0.00000567 seconds
- P95 inference latency per sample: 0.00000767 seconds
- Device used: cpu
- Runtime: 10.00 seconds

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

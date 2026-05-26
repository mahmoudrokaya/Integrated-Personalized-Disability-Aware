# Scientific Audit Report

## 1. Dataset overview
The dataset contains **10499 rows** and **34 columns**, of which **30** are candidate predictors after excluding the administrative fields and the target. The analysis is centered on `disability_stage`, and the initial high-value baseline set is: **age, max_grip_strength, BMI, MET_min_week, PHQ9_score, medication_count, HbA1c, mean_SBP**.

## 2. How the data should be read logically
The first reading layer is logical rather than numerical. Each factor is interpreted according to domain meaning, historical relation to disability, and whether it is directly measured, derived, or potentially redundant. This prevents the model from treating all columns as equally independent scientific entities.

## 3. How the data should be read quantitatively
The second reading layer asks what the actual numbers say. For numeric variables, the script examines distribution, missingness, by-stage summaries, and monotonic association with disability stage. For categorical variables, it examines distributional shifts across stages. The goal is to compare expected facts with observed facts.

## 4. Key dependency checks
- **BMI**: BMI is calculated from height and weight. Rows with |BMI - recalculated BMI| > 2 = 0.
- **LDL**: LDL is linked to other lipid measures and often formula-derived. Rows with |LDL - formula| > 20 = 0; LDL > total cholesterol rows = 38.
- **PHQ9_category**: PHQ9 category is derived from PHQ9 score. PHQ9 category mismatches = 248.
- **meets_activity_guideline**: Guideline attainment is derived from weekly MET-minutes. Guideline mismatches = 0.
- **polypharmacy**: Polypharmacy is derived from medication count threshold. Polypharmacy mismatches = 0.

## 5. Strongest observed associations (screening view)
- **mobility_difficulty**: spearman_rho_vs_disability_stage = 0.8559
- **adl_difficulty**: spearman_rho_vs_disability_stage = 0.7411
- **upper_limb_difficulty**: spearman_rho_vs_disability_stage = 0.6286
- **PHQ9_score**: spearman_rho_vs_disability_stage = 0.3358
- **medication_count**: spearman_rho_vs_disability_stage = 0.3134
- **arthritis**: spearman_rho_vs_disability_stage = 0.2998
- **polypharmacy**: spearman_rho_vs_disability_stage = 0.286
- **PHQ9_category**: spearman_rho_vs_disability_stage = 0.2805
- **BMI**: spearman_rho_vs_disability_stage = 0.1841
- **income_poverty_ratio**: spearman_rho_vs_disability_stage = -0.1475

## 6. Potential red flags
### Variable-level red flags
- **BMI**: BMI_out_of_expected_range
### Known assumptions not clearly confirmed
- **age**: expected `positive` relation, observed `0.0028`, verdict = **not_confirmed**.
- **mean_SBP**: expected `positive` relation, observed `0.0034`, verdict = **not_confirmed**.

## 7. Recommended modeling sequence
Start with the high-value set. Then add demographic/social variables, cardiometabolic expansion variables, behavioral expansion variables, structural disease variables, and finally derived variables or the full model. At each stage, ask whether the added group improves predictive value, confirms expected mechanisms, or only duplicates information already present.

## 8. Practical conclusion
This audit is intended to answer whether the dataset behaves like real clinical and biological data before heavy modeling begins. If the observed numbers agree with known facts, modeling can proceed with confidence. If expected patterns fail or derived variables behave inconsistently, those issues should be investigated before final model selection.

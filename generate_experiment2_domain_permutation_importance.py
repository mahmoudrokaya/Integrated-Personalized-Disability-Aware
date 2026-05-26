import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path(
    r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\Experiment2\Results"
)

INPUT_FILE = RESULTS_DIR / "experiment2_integrated_team_explainability_summary.csv"

OUTPUT_FIGURE = RESULTS_DIR / "experiment2_domain_permutation_importance.png"

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

# ============================================================
# Prepare plotting values
# ============================================================

teams = df["team"]

accuracy_drop = df["permutation_accuracy_drop"]

f1_drop = df["permutation_f1_weighted_drop"]

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(12, 7))

x = range(len(teams))
width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    accuracy_drop,
    width=width,
    label="Accuracy Drop",
)

plt.bar(
    [i + width / 2 for i in x],
    f1_drop,
    width=width,
    label="Weighted F1 Drop",
)

plt.xticks(x, teams, rotation=20)

plt.ylabel("Performance Degradation")

plt.xlabel("Specialist Organizational Team")

plt.title(
    "Domain-Level Permutation Importance Analysis"
)

plt.legend()

plt.tight_layout()

# ============================================================
# Save
# ============================================================

plt.savefig(OUTPUT_FIGURE, dpi=600, bbox_inches="tight")

print("\nFigure generated successfully.")
print(f"Saved to:\n{OUTPUT_FIGURE}")
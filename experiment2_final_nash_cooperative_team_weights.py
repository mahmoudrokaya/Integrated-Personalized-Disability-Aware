import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path(
    r"D:\47\472\New-Papers\Integrated Personalized Disability-Aware\Experiments\Experiment2\Results"
)

INPUT_FILE = RESULTS_DIR / "team_cooperative_weights.csv"

OUTPUT_FIGURE = RESULTS_DIR / "experiment2_final_nash_cooperative_team_weights.png"

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Expected columns:
# team, cooperative_weight

df = df.sort_values("cooperative_weight", ascending=False)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(11, 7))

bars = plt.bar(
    df["team"],
    df["cooperative_weight"]
)

plt.ylabel("Final cooperative weight")
plt.xlabel("Specialist organizational team")
plt.title("Final Nash-Cooperative Specialist-Team Weighting Distribution")

plt.xticks(rotation=25, ha="right")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()

# ============================================================
# Save
# ============================================================

plt.savefig(OUTPUT_FIGURE, dpi=600, bbox_inches="tight")
plt.close()

print("Figure 9 generated successfully.")
print(f"Saved to: {OUTPUT_FIGURE}")
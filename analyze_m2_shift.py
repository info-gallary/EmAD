"""
Mission 2 Temporal Distribution Shift Analysis
Shows how class balance changes over time, explaining the 46% accuracy
under chronological splits.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

CSV   = r"d:\UbtVM-Def\Models\data\mission2_preprocessed.csv"
OUT   = r"d:\UbtVM-Def\Models\reports\missions\m2\m2_distribution_shift.png"
OUT2  = r"d:\UbtVM-Def\Models\reports\missions\m2\m2_class_timeline.png"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.35, "grid.linestyle": "--",
})

df = pd.read_csv(CSV, index_col=0)
labels = df["label"].values
n = len(labels)

# ── Figure 1: rolling class proportion over time ─────────────────────────────
window = 500  # ~8 hours at 60s sampling
classes = sorted(set(labels.tolist()))
class_names = {0: "Normal", 5: "Rare-Event"}
colors = {0: "#2196F3", 5: "#FF5722"}

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# top panel: raw label as scatter
ax = axes[0]
for cls in classes:
    idx = np.where(labels == cls)[0]
    ax.scatter(idx, np.ones(len(idx)) * cls, c=colors.get(cls, "gray"),
               s=1, alpha=0.4, label=class_names.get(cls, str(cls)))
ax.set_yticks(classes)
ax.set_yticklabels([class_names.get(c, str(c)) for c in classes])
ax.set_ylabel("Class")
ax.set_title("Mission 2 — Raw Label Timeline")
ax.legend(markerscale=6, loc="upper right")

# train/val/test boundary lines
n_tr = int(0.70 * n); n_va = int(0.85 * n)
for a in axes:
    a.axvline(n_tr, color="green",  ls="--", lw=1.5, label="Train end (70%)")
    a.axvline(n_va, color="orange", ls="--", lw=1.5, label="Val end (85%)")

# bottom panel: rolling proportion of Normal
ax2 = axes[1]
roll_normal = pd.Series((labels == 0).astype(float)).rolling(window, center=True).mean()
roll_rare   = pd.Series((labels == 5).astype(float)).rolling(window, center=True).mean()
x = np.arange(n)
ax2.fill_between(x, 0, roll_normal, alpha=0.5, color="#2196F3", label="Normal")
ax2.fill_between(x, roll_normal, roll_normal + roll_rare, alpha=0.5, color="#FF5722", label="Rare-Event")
ax2.set_ylim(0, 1); ax2.set_ylabel("Rolling Proportion")
ax2.set_xlabel("Timestamp Index")
ax2.set_title(f"Mission 2 — Rolling Class Proportion (window={window} steps)")
ax2.legend(loc="upper left")

# annotate percentages in each split region
for region, start, end, name in [
    ("Train\n(70%)", 0, n_tr, "train"),
    ("Val\n(15%)", n_tr, n_va, "val"),
    ("Test\n(15%)", n_va, n, "test"),
]:
    seg = labels[start:end]
    pn = (seg == 0).mean() * 100
    pr = (seg == 5).mean() * 100
    mid = (start + end) / 2
    ax2.text(mid, 0.5, f"Normal\n{pn:.0f}%\nRare\n{pr:.0f}%",
             ha="center", va="center", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.suptitle("Mission 2: Temporal Distribution Shift\n"
             "Class balance inverts between training and test portions",
             fontweight="bold", y=1.01)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}")

# ── Figure 2: bar chart comparing split proportions ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
splits   = ["Train (70%)", "Validation (15%)", "Test (15%)"]
starts   = [0, n_tr, n_va]
ends     = [n_tr, n_va, n]
normals  = [(labels[s:e] == 0).mean() * 100 for s, e in zip(starts, ends)]
rares    = [(labels[s:e] == 5).mean() * 100 for s, e in zip(starts, ends)]

x = np.arange(len(splits)); w = 0.35
b1 = ax.bar(x - w/2, normals, w, label="Normal",     color="#2196F3", alpha=0.85)
b2 = ax.bar(x + w/2, rares,   w, label="Rare-Event", color="#FF5722", alpha=0.85)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(splits)
ax.set_ylabel("Class Proportion (%)"); ax.set_ylim(0, 110)
ax.set_title("Mission 2: Class Balance Per Split\n"
             "Train is 85% Rare-Event; Test is 54% Normal — model fails on test")
ax.legend()
ax.axhline(50, color="gray", ls=":", lw=1, alpha=0.7, label="50% line")

# print exact stats
print("\nMission 2 class distribution per split:")
for name, s, e in zip(splits, starts, ends):
    seg = labels[s:e]
    print(f"  {name}: Normal={( seg==0).mean()*100:.1f}%  Rare-Event={(seg==5).mean()*100:.1f}%  (n={len(seg):,})")

plt.tight_layout()
plt.savefig(OUT2, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT2}")

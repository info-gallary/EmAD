"""Generate m2_class_timeline.png showing the temporal class-ratio shift on Mission 2."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV   = r"d:\UbtVM-Def\Models\data\mission2_preprocessed.csv"
OUT   = r"d:\UbtVM-Def\Models\reports\missions\m2\m2_class_timeline.png"

df = pd.read_csv(CSV, parse_dates=["timestamp"], usecols=["timestamp", "label", "class_name"])
df = df.sort_values("timestamp").reset_index(drop=True)
n  = len(df)
print(f"M2 rows: {n}  span: {df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]}")

# Per-class chronological split markers (70/15/15)
split_train_val_idx = int(n * 0.70)
split_val_test_idx  = int(n * 0.85)
t_train_val = df.timestamp.iloc[split_train_val_idx]
t_val_test  = df.timestamp.iloc[split_val_test_idx]

# Bin into ~50 windows and compute Rare-Event fraction per bin
df["bin"] = pd.cut(df.index, bins=50, labels=False)
rare_label = df.loc[df["class_name"] != "Normal", "label"].iloc[0]
agg = df.groupby("bin").agg(
    t_start=("timestamp", "min"),
    t_mid=("timestamp", lambda s: s.iloc[len(s)//2]),
    rare_frac=("label", lambda s: float((s == rare_label).mean())),
    n=("label", "size"),
).reset_index()

fig, ax = plt.subplots(figsize=(11, 5))

# Background shading for train / val / test
ax.axvspan(df.timestamp.iloc[0],  t_train_val, alpha=0.08, color="#1976D2", label="Train (70%)")
ax.axvspan(t_train_val,           t_val_test,  alpha=0.10, color="#FB8C00", label="Validation (15%)")
ax.axvspan(t_val_test,            df.timestamp.iloc[-1], alpha=0.18, color="#C62828", label="Test (15%)")

# Rare-Event fraction line
ax.plot(agg.t_mid, agg.rare_frac * 100.0, color="#1a1a1a", lw=2.0, marker="o", ms=3.5,
        label="Rare-Event window fraction (%)")

# Mean Rare-Event % per split
train_pct = df.iloc[:split_train_val_idx]["label"].eq(rare_label).mean() * 100
val_pct   = df.iloc[split_train_val_idx:split_val_test_idx]["label"].eq(rare_label).mean() * 100
test_pct  = df.iloc[split_val_test_idx:]["label"].eq(rare_label).mean() * 100

# Annotate per-split means
def midpoint(a, b):
    return a + (b - a) / 2

ax.annotate(f"Train mean: {train_pct:.1f}%",
            xy=(midpoint(df.timestamp.iloc[0], t_train_val), 92),
            ha="center", fontsize=10, color="#1976D2",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1976D2", lw=1.0))
ax.annotate(f"Val mean: {val_pct:.1f}%",
            xy=(midpoint(t_train_val, t_val_test), 92),
            ha="center", fontsize=10, color="#FB8C00",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#FB8C00", lw=1.0))
ax.annotate(f"Test mean: {test_pct:.1f}%",
            xy=(midpoint(t_val_test, df.timestamp.iloc[-1]), 92),
            ha="center", fontsize=10, color="#C62828",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C62828", lw=1.0))

# Drift ratio annotation
shift = test_pct / max(train_pct, 0.1)
ax.text(0.02, 0.96,
        f"Distribution shift: test Rare-Event fraction is {shift:.1f}x training\n"
        f"(local-feature models collapse to ~35%; Transformer recovers to 76.79%)",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FFF59D", ec="#F57F17", lw=1.0))

ax.set_xlabel("Mission timeline", fontsize=12)
ax.set_ylabel("Rare-Event fraction per bin (%)", fontsize=12)
ax.set_title("Mission 2 — Temporal Class-Ratio Shift (Chronological Splits)",
             fontsize=13, fontweight="bold")
ax.set_ylim(-2, 105)
ax.grid(True, alpha=0.25)
ax.legend(loc="center right", fontsize=9, framealpha=0.95)

ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT}")
print(f"Train Rare-Event %: {train_pct:.2f}")
print(f"Val   Rare-Event %: {val_pct:.2f}")
print(f"Test  Rare-Event %: {test_pct:.2f}")
print(f"Shift ratio (test / train): {shift:.2f}x")

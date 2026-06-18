"""
Reviewer Point 2 figure: seed stability of default-threshold vs calibrated accuracy.

Reads reports/revision/multiseed_results.json (3 seeds x 4 deep models x 3 missions)
and renders the headline M2 result: under covariate drift the DEFAULT-threshold
accuracy is a near-random function of initialisation (std up to 27 pp), whereas the
prior-matched post-hoc CALIBRATED accuracy is stable and SOTA-level across both seeds
AND architectures (95 +/- ~1 pp), matching/beating RandomForest (95.67 %).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"d:\UbtVM-Def\Models\reports\revision"
RES = json.load(open(os.path.join(OUT, "multiseed_results.json")))["results"]
RF_M2 = 95.67  # RandomForest, identical split (Comment 17)

models = ["CNN", "BiLSTM", "Transformer", "ConvFormer"]
m2 = RES["mission2"]
raw_m = [m2[k]["accuracy"]["mean"] for k in models]
raw_s = [m2[k]["accuracy"]["std"] for k in models]
cal_m = [m2[k]["calibrated_accuracy"]["mean"] for k in models]
cal_s = [m2[k]["calibrated_accuracy"]["std"] for k in models]

fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(models)); w = 0.38
b1 = ax.bar(x - w/2, raw_m, w, yerr=raw_s, capsize=5, color="#c0504d",
            label="Default threshold (τ=0.5)", edgecolor="black", linewidth=0.6)
b2 = ax.bar(x + w/2, cal_m, w, yerr=cal_s, capsize=5, color="#4f81bd",
            label="Prior-matched calibration", edgecolor="black", linewidth=0.6)
ax.axhline(RF_M2, ls="--", lw=1.4, color="#2e7d32")
ax.text(len(models) - 0.5, RF_M2 + 1.0, f"RandomForest {RF_M2:.1f}%",
        color="#2e7d32", ha="right", va="bottom", fontsize=9, fontweight="bold")

for xi, (m, s) in zip(x - w/2, zip(raw_m, raw_s)):
    ax.text(xi, m + s + 1.2, f"{m:.0f}±{s:.0f}", ha="center", va="bottom",
            fontsize=8.5, color="#7a2a28")
for xi, (m, s) in zip(x + w/2, zip(cal_m, cal_s)):
    ax.text(xi, m + s + 1.2, f"{m:.1f}±{s:.1f}", ha="center", va="bottom",
            fontsize=8.5, color="#2a4a7a", fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, 110)
ax.set_title("Mission 2 (covariate drift): default-threshold accuracy is seed-unstable;\n"
             "post-hoc calibration is stable and SOTA across seeds and architectures\n"
             "(mean ± std over seeds {42, 3, 7})", fontsize=10.5)
ax.legend(loc="lower center", ncol=2, frameon=True, fontsize=9)
ax.grid(axis="y", ls=":", alpha=0.45)
plt.tight_layout()
p = os.path.join(OUT, "seed_variance_m2.png")
plt.savefig(p, dpi=200); plt.close()
print("Saved", p)

# Console summary used for the REVIEWER_RESPONSE table
for mid in ("mission2", "mission3", "mission1"):
    print(f"\n[{mid}]")
    for k in models:
        r = RES[mid][k]
        a = r["accuracy"]
        s = f"  {k:12s} raw {a['mean']:6.2f}±{a['std']:5.2f}"
        if "calibrated_accuracy" in r:
            c = r["calibrated_accuracy"]
            s += f"   cal {c['mean']:6.2f}±{c['std']:5.2f}"
        print(s)

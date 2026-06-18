"""
Figure: Mission-2 accuracy before vs after post-hoc threshold calibration,
with the RandomForest classical baseline as the SOTA reference line.
Shows that well-ranked deep models (high ROC-AUC) recover to / beyond SOTA.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"d:\UbtVM-Def\Models\reports\revision"
cal = json.load(open(os.path.join(OUT, "calibration.json")))
em  = json.load(open(os.path.join(OUT, "expanded_metrics.json")))

models = ["CNN", "BiLSTM", "Transformer", "ConvFormer", "Hybrid"]
m2 = cal["mission2"]["models"]
base = [m2[m]["baseline_tau0.5"]["accuracy"] for m in models]
valf1 = [m2[m]["val_macro_f1"]["accuracy"] for m in models]
prior = [m2[m]["prior_match"]["accuracy"] for m in models]
auc  = [em["mission2"]["models"][m]["roc_auc"] for m in models]
RF_SOTA = 95.67  # RandomForest M2 (best classical baseline)

x = np.arange(len(models)); w = 0.26
fig, ax = plt.subplots(figsize=(10, 5.5))
b1 = ax.bar(x - w, base, w, label="Baseline (τ=0.5)", color="#C62828")
b2 = ax.bar(x, valf1, w, label="Calibrated (val-tuned, supervised)", color="#FB8C00")
b3 = ax.bar(x + w, prior, w, label="Calibrated (prior-match, unsupervised)", color="#2E7D32")
ax.axhline(RF_SOTA, ls="--", lw=1.8, color="#1565C0",
           label=f"RandomForest baseline ({RF_SOTA}%)")
for xi, a in zip(x, auc):
    ax.text(xi, 3, f"AUC\n{a:.2f}", ha="center", va="bottom", fontsize=8, color="#333")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 105)
ax.set_title("Mission 2: post-hoc threshold calibration recovers deep models to SOTA\n"
             "(models with high ROC-AUC were mis-thresholded, not mis-trained)",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "calibration_m2.png"), dpi=300, bbox_inches="tight")
print("Saved: calibration_m2.png")

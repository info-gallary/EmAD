"""
generate_pub_figures.py
Run AFTER train_all_missions.py completes.
Generates publication-quality figures and tables for top-tier submission.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import re
import warnings
warnings.filterwarnings("ignore")

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

REPORT_DIR  = r"d:\UbtVM-Def\Models\reports\missions"
GEN_DIR     = r"d:\UbtVM-Def\Models\reports\generalized"
OUT_DIR     = r"d:\UbtVM-Def\Models\reports\publication"
MULTISEED_F = os.path.join(REPORT_DIR, "multiseed_summary.txt")

MODELS   = ["CNN", "BiLSTM", "Transformer", "ConvFormer", "VAE", "Hybrid"]
MISSIONS = [1, 2, 3]
PALETTE  = {"CNN": "#1565C0", "BiLSTM": "#6A1B9A", "Transformer": "#00695C",
            "ConvFormer": "#E65100", "VAE": "#2E7D32", "Hybrid": "#880E4F"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "axes.titleweight": "bold", "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "lines.linewidth": 2.0,
})

os.makedirs(OUT_DIR, exist_ok=True)


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_mission_report(mid):
    path = os.path.join(REPORT_DIR, f"m{mid}", f"m{mid}_report.txt")
    results = {}
    if not os.path.exists(path):
        return results
    with open(path, encoding="utf-8") as f:
        text = f.read()
    pat = r'^\s+(CNN|BiLSTM|Transformer|ConvFormer|VAE|Hybrid)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    for m in re.finditer(pat, text, re.MULTILINE):
        results[m.group(1)] = {
            "acc":  float(m.group(2)) / 100,
            "f1":   float(m.group(3)),
            "prec": float(m.group(4)),
            "rec":  float(m.group(5)),
        }
    return results


def parse_multiseed():
    """Returns {mid: {model: {"mean_acc", "std_acc", "mean_f1", "std_f1"}}}"""
    if not os.path.exists(MULTISEED_F):
        return {}
    with open(MULTISEED_F, encoding="utf-8") as f:
        text = f.read()
    out = {}
    cur_mid = None
    for line in text.splitlines():
        m_mid = re.search(r'Mission\s+(\d+)', line)
        if m_mid:
            cur_mid = int(m_mid.group(1))
            out[cur_mid] = {}
            continue
        pat = r'(CNN|BiLSTM|Transformer|ConvFormer|VAE|Hybrid)\s+([\d.]+)\s*[±\?]\s*([\d.]+)%\s+([\d.]+)\s*[±\?]\s*([\d.]+)'
        m = re.search(pat, line)
        if m and cur_mid is not None:
            out[cur_mid][m.group(1)] = {
                "mean_acc": float(m.group(2)) / 100,
                "std_acc":  float(m.group(3)) / 100,
                "mean_f1":  float(m.group(4)),
                "std_f1":   float(m.group(5)),
            }
    return out


# ── Figure 1: Comprehensive model comparison (2×2 metrics, all missions) ─────

def fig_comprehensive_comparison(single_seed, multiseed, out):
    metrics = [("acc", "Accuracy (%)"), ("f1", "Weighted F1"),
               ("prec", "Precision"),   ("rec", "Recall")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for ax, (met, label) in zip(axes, metrics):
        n_m = len(MODELS); w = 0.75 / n_m
        x   = np.arange(len(MISSIONS))
        for i, model in enumerate(MODELS):
            vals, errs = [], []
            for mid in MISSIONS:
                key = f"mean_{met}"
                ms_has = multiseed and mid in multiseed and model in multiseed[mid] and key in multiseed[mid][model]
                ss_has = mid in single_seed and model in single_seed[mid]
                if ms_has:
                    v = multiseed[mid][model][key]
                    e = multiseed[mid][model].get(f"std_{met}", 0)
                    if met in ("acc", "prec", "rec"):
                        v *= 100; e *= 100
                    vals.append(v); errs.append(e)
                elif ss_has:
                    v = single_seed[mid][model][met]
                    if met in ("acc", "prec", "rec"):
                        v *= 100
                    vals.append(v); errs.append(0)
                else:
                    vals.append(0); errs.append(0)

            offset = (i - n_m / 2 + 0.5) * w
            bars = ax.bar(x + offset, vals, w, yerr=errs if any(e > 0 for e in errs) else None,
                         label=model, color=PALETTE[model], alpha=0.87,
                         capsize=3, error_kw={"linewidth": 1.2})
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(errs) if errs else 0) + 0.5,
                            f"{v:.1f}" if met in ("acc","prec","rec") else f"{v:.3f}",
                            ha="center", va="bottom", fontsize=6.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"Mission {m}" for m in MISSIONS])
        ax.set_ylabel(label)
        ax.set_title(label)
        ymax = 108 if met in ("acc","prec","rec") else 1.08
        ax.set_ylim(0, ymax)
        if met in ("acc","prec","rec"):
            ax.axhline(95, color="red", ls="--", lw=1, alpha=0.6, label="95% line")
        ax.legend(fontsize=8, ncol=2)

    seed_note = f"Mean ± Std (3 seeds)" if multiseed else "Single seed (seed=42)"
    plt.suptitle(f"EmAD: All-Models Cross-Mission Performance\n({seed_note})",
                 fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 2: Accuracy heatmap ────────────────────────────────────────────────

def fig_heatmap(single_seed, multiseed, out):
    models_ordered = MODELS
    data = np.zeros((len(models_ordered), len(MISSIONS)))
    for j, mid in enumerate(MISSIONS):
        for i, model in enumerate(models_ordered):
            if multiseed and mid in multiseed and model in multiseed[mid]:
                data[i, j] = multiseed[mid][model]["mean_acc"] * 100
            elif mid in single_seed and model in single_seed[mid]:
                data[i, j] = single_seed[mid][model]["acc"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = "RdYlGn"
    im = ax.imshow(data, cmap=cmap, vmin=30, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")

    ax.set_xticks(range(len(MISSIONS)))
    ax.set_xticklabels([f"Mission {m}" for m in MISSIONS], fontsize=11)
    ax.set_yticks(range(len(models_ordered)))
    ax.set_yticklabels(models_ordered, fontsize=11)

    for i in range(len(models_ordered)):
        for j in range(len(MISSIONS)):
            v = data[i, j]
            color = "white" if v < 60 else "black"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_title("Model Accuracy Heatmap — All Missions", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 3: F1 radar / spider chart ────────────────────────────────────────

def fig_radar(single_seed, multiseed, out):
    categories = [f"M{m}" for m in MISSIONS]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)

    for model in MODELS:
        vals = []
        for mid in MISSIONS:
            if multiseed and mid in multiseed and model in multiseed[mid]:
                vals.append(multiseed[mid][model]["mean_acc"] * 100)
            elif mid in single_seed and model in single_seed[mid]:
                vals.append(single_seed[mid][model]["acc"] * 100)
            else:
                vals.append(0)
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, label=model, color=PALETTE[model])
        ax.fill(angles, vals, alpha=0.08, color=PALETTE[model])

    ax.set_title("Model Accuracy Radar — All Missions", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 4: Per-mission best-model confusion matrices (panel) ───────────────

def fig_cm_panel(out):
    import matplotlib.image as mpimg
    best_models = {1: "cnn", 2: "transformer", 3: "transformer"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, mid in zip(axes, MISSIONS):
        model = best_models[mid]
        path  = os.path.join(REPORT_DIR, f"m{mid}", f"{model}_confusion_matrix.png")
        if os.path.exists(path):
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Mission {mid} — Best Model ({model.upper()})", fontweight="bold")
        else:
            ax.text(0.5, 0.5, f"Missing:\n{path}", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
    plt.suptitle("Confusion Matrices — Best Model Per Mission", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Figure 5: McNemar's statistical significance test ────────────────────────

def fig_mcnemar(out):
    """
    McNemar's test between each pair of models on each mission.
    Requires per-sample prediction files — reads confusion matrix images only if
    raw predictions not available; falls back to a note.
    Reads stored numpy prediction arrays if present.
    """
    import glob
    pred_files = glob.glob(os.path.join(REPORT_DIR, "m*/predictions_*.npz"))
    if not pred_files:
        print("  [McNemar] No .npz prediction files found — skipping (run after training saves predictions)")
        return

    try:
        from statsmodels.stats.contingency_tables import mcnemar as mc_test
    except ImportError:
        print("  [McNemar] statsmodels not installed — skipping fig5")
        return

    results = {}
    for mid in MISSIONS:
        preds = {}
        true_labels = None
        for model in MODELS:
            f = os.path.join(REPORT_DIR, f"m{mid}", f"predictions_{model.lower()}.npz")
            if os.path.exists(f):
                d = np.load(f)
                preds[model] = d["pred"]
                if true_labels is None:
                    true_labels = d["true"]
        if len(preds) < 2 or true_labels is None:
            continue

        model_list = sorted(preds.keys())
        n = len(model_list)
        p_matrix = np.ones((n, n))
        for i, m1 in enumerate(model_list):
            for j, m2 in enumerate(model_list):
                if i >= j:
                    continue
                correct1 = (preds[m1] == true_labels).astype(int)
                correct2 = (preds[m2] == true_labels).astype(int)
                b = ((correct1 == 1) & (correct2 == 0)).sum()
                c = ((correct1 == 0) & (correct2 == 1)).sum()
                if b + c == 0:
                    p = 1.0
                else:
                    from scipy.stats import binom
                    p = 2 * min(binom.cdf(min(b, c), b + c, 0.5),
                                1 - binom.cdf(min(b, c) - 1, b + c, 0.5))
                p_matrix[i, j] = p_matrix[j, i] = p
        results[mid] = (model_list, p_matrix)

    if not results:
        return

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, (mid, (ml, pm)) in zip(axes, results.items()):
        mask = np.eye(len(ml), dtype=bool)
        pm_display = np.where(mask, np.nan, pm)
        im = ax.imshow(pm_display, cmap="RdYlGn_r", vmin=0, vmax=0.1)
        ax.set_xticks(range(len(ml))); ax.set_xticklabels(ml, rotation=45, ha="right")
        ax.set_yticks(range(len(ml))); ax.set_yticklabels(ml)
        for i in range(len(ml)):
            for j in range(len(ml)):
                if i != j:
                    sig = "***" if pm[i,j] < 0.001 else ("**" if pm[i,j] < 0.01 else ("*" if pm[i,j] < 0.05 else "ns"))
                    ax.text(j, i, f"{pm[i,j]:.3f}\n{sig}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, label="p-value")
        ax.set_title(f"Mission {mid}\nMcNemar p-values (* p<.05)")
    plt.suptitle("Statistical Significance: McNemar's Test Between Models", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── LaTeX table ───────────────────────────────────────────────────────────────

def latex_table(single_seed, multiseed):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Model performance across ESA missions (Accuracy \% / Weighted F1). "
        r"Best result per mission in \textbf{bold}.}",
        r"\label{tab:results}",
        r"\begin{tabular}{l|cc|cc|cc}",
        r"\hline",
        r"\textbf{Model} & \multicolumn{2}{c|}{\textbf{Mission 1}} & "
        r"\multicolumn{2}{c|}{\textbf{Mission 2}} & \multicolumn{2}{c}{\textbf{Mission 3}} \\",
        r" & Acc\% & F1 & Acc\% & F1 & Acc\% & F1 \\",
        r"\hline",
    ]

    # collect all values to find best per mission
    best_acc = {}
    for mid in MISSIONS:
        best_acc[mid] = 0
        for model in MODELS:
            if multiseed and mid in multiseed and model in multiseed[mid]:
                v = multiseed[mid][model]["mean_acc"] * 100
            elif mid in single_seed and model in single_seed[mid]:
                v = single_seed[mid][model]["acc"] * 100
            else:
                v = 0
            best_acc[mid] = max(best_acc[mid], v)

    for model in MODELS:
        row = [model.replace("BiLSTM", "BiLSTM").replace("Transformer", "Transformer")]
        for mid in MISSIONS:
            if multiseed and mid in multiseed and model in multiseed[mid]:
                acc = multiseed[mid][model]["mean_acc"] * 100
                std = multiseed[mid][model]["std_acc"]  * 100
                f1  = multiseed[mid][model]["mean_f1"]
                acc_str = f"{acc:.1f}$\\pm${std:.1f}"
                f1_str  = f"{f1:.3f}"
            elif mid in single_seed and model in single_seed[mid]:
                acc = single_seed[mid][model]["acc"] * 100
                f1  = single_seed[mid][model]["f1"]
                acc_str = f"{acc:.1f}"
                f1_str  = f"{f1:.3f}"
            else:
                acc_str = "--"; f1_str = "--"; acc = 0

            if acc >= best_acc[mid] - 0.01:
                acc_str = f"\\textbf{{{acc_str}}}"
                f1_str  = f"\\textbf{{{f1_str}}}"
            row.append(acc_str)
            row.append(f1_str)
        lines.append(" & ".join(row) + r" \\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}\n  Publication Figure Generator\n{'='*60}")

    # load data
    single_seed = {mid: parse_mission_report(mid) for mid in MISSIONS}
    multiseed   = parse_multiseed()

    if multiseed:
        print(f"  Multi-seed data found for missions: {sorted(multiseed.keys())}")
    else:
        print("  No multi-seed summary yet — using single-seed (seed=42) results")

    for mid in MISSIONS:
        if single_seed[mid]:
            print(f"  Mission {mid}: {list(single_seed[mid].keys())}")
        else:
            print(f"  Mission {mid}: no report found")

    # generate figures
    fig_comprehensive_comparison(single_seed, multiseed,
                                  os.path.join(OUT_DIR, "fig1_model_comparison.png"))
    fig_heatmap(single_seed, multiseed,
                os.path.join(OUT_DIR, "fig2_accuracy_heatmap.png"))
    fig_radar(single_seed, multiseed,
              os.path.join(OUT_DIR, "fig3_radar.png"))
    fig_cm_panel(os.path.join(OUT_DIR, "fig4_confusion_panel.png"))
    fig_mcnemar(os.path.join(OUT_DIR, "fig5_mcnemar.png"))

    # LaTeX table
    table = latex_table(single_seed, multiseed)
    tbl_path = os.path.join(OUT_DIR, "table1_results.tex")
    with open(tbl_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"  Saved: {tbl_path}")
    print(f"\n  LaTeX Table:\n{table}\n")

    # text summary
    print(f"\n{'='*60}\n  RESULTS SUMMARY\n{'='*60}")
    for mid in MISSIONS:
        print(f"\n  Mission {mid}:")
        for model in MODELS:
            if mid in single_seed and model in single_seed[mid]:
                r = single_seed[mid][model]
                print(f"    {model:<14} Acc={r['acc']*100:.2f}%  F1={r['f1']:.4f}  Prec={r['prec']:.4f}  Rec={r['rec']:.4f}")

    print(f"\n  All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

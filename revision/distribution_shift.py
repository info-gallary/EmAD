"""
Reviewer Point 11: deeper investigation of train-test distribution shift.

Quantifies the shift between TRAIN and TEST feature distributions for each
mission using four complementary measures:
  - Kolmogorov-Smirnov statistic (per feature, averaged)
  - Wasserstein-1 distance (per feature, averaged)
  - Population Stability Index (PSI) — standard drift metric in ML monitoring
  - Jensen-Shannon divergence (symmetric, bounded KL)

Also quantifies LABEL shift (class-prior change) which is the dominant effect on M2.
Produces a per-mission summary + a comparison figure.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance

import sys
sys.path.insert(0, r"d:\UbtVM-Def\Models")
import train_all_missions as T

OUT = r"d:\UbtVM-Def\Models\reports\revision"
os.makedirs(OUT, exist_ok=True)


def psi(expected, actual, bins=10):
    """Population Stability Index between two 1D samples."""
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / max(len(expected), 1) + 1e-6
    a = np.histogram(actual, qs)[0] / max(len(actual), 1) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def js_divergence(p, q, bins=30):
    lo = min(p.min(), q.min()); hi = max(p.max(), q.max())
    edges = np.linspace(lo, hi, bins + 1)
    P = np.histogram(p, edges)[0] / max(len(p), 1) + 1e-9
    Q = np.histogram(q, edges)[0] / max(len(q), 1) + 1e-9
    P /= P.sum(); Q /= Q.sum()
    M = 0.5 * (P + Q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return float(0.5 * kl(P, M) + 0.5 * kl(Q, M))


def load_windows(mid):
    csv = os.path.join(T.DATA_DIR, f"mission{mid}_preprocessed.csv")
    df = pd.read_csv(csv, index_col=0)
    fcols = [c for c in df.columns if c not in ("label", "class_name", "mission_id")]
    X = df[fcols].values.astype(np.float32)
    y_raw = df["label"].values.astype(np.int64)
    uniq = sorted(set(y_raw.tolist()))
    remap = {o: n for n, o in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    Xw, yw = T.make_windows(X, y, T.WINDOW, T.STEP)
    Xtr, ytr, Xva, yva, Xte, yte = T.per_class_chron_split(Xw, yw)
    # window-level feature = per-channel mean over the window
    return Xtr.mean(1), ytr, Xte.mean(1), yte, fcols


def main():
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mid in zip(axes, (1, 2, 3)):
        Ftr, ytr, Fte, yte, fcols = load_windows(mid)
        nfeat = Ftr.shape[1]
        ks_vals, w_vals, psi_vals, js_vals = [], [], [], []
        for k in range(nfeat):
            a, b = Ftr[:, k], Fte[:, k]
            ks_vals.append(ks_2samp(a, b).statistic)
            w_vals.append(wasserstein_distance(a, b))
            psi_vals.append(psi(a, b))
            js_vals.append(js_divergence(a, b))
        # label (prior) shift
        def prior(y, n):
            c = np.bincount(y, minlength=n) / len(y); return c
        n_cls = int(max(ytr.max(), yte.max()) + 1)
        ptr, pte = prior(ytr, n_cls), prior(yte, n_cls)
        label_js = js_divergence_discrete(ptr, pte)
        summary[f"mission{mid}"] = {
            "n_features": nfeat,
            "covariate_shift": {
                "ks_mean": float(np.mean(ks_vals)), "ks_max": float(np.max(ks_vals)),
                "wasserstein_mean": float(np.mean(w_vals)),
                "psi_mean": float(np.mean(psi_vals)), "psi_max": float(np.max(psi_vals)),
                "js_mean": float(np.mean(js_vals)), "js_max": float(np.max(js_vals)),
                "frac_features_psi_gt_0.25": float(np.mean(np.array(psi_vals) > 0.25)),
            },
            "label_shift": {
                "train_prior": [float(x) for x in ptr],
                "test_prior": [float(x) for x in pte],
                "js_divergence": float(label_js),
                "anomaly_ratio_train": float(1 - ptr[0]),
                "anomaly_ratio_test": float(1 - pte[0]),
                "ratio_shift_factor": float((1 - pte[0]) / max(1 - ptr[0], 1e-6)),
            },
        }
        # plot PSI distribution per mission
        ax.hist(psi_vals, bins=30, color="#1976D2", alpha=0.8)
        ax.axvline(0.25, color="#C62828", ls="--", lw=1.5, label="PSI=0.25 (moderate)")
        ax.axvline(0.10, color="#FB8C00", ls=":", lw=1.5, label="PSI=0.10 (minor)")
        ax.set_title(f"Mission {mid}  (mean PSI={np.mean(psi_vals):.3f})")
        ax.set_xlabel("Per-feature PSI (train vs test)"); ax.set_ylabel("# features")
        ax.legend(fontsize=8)
        print(f"M{mid}: KS={np.mean(ks_vals):.3f} Wass={np.mean(w_vals):.4f} "
              f"PSI={np.mean(psi_vals):.3f} JS={np.mean(js_vals):.4f} | "
              f"label-shift factor={summary[f'mission{mid}']['label_shift']['ratio_shift_factor']:.2f}x "
              f"(anom {summary[f'mission{mid}']['label_shift']['anomaly_ratio_train']*100:.1f}%->"
              f"{summary[f'mission{mid}']['label_shift']['anomaly_ratio_test']*100:.1f}%)")
    plt.suptitle("Covariate Shift (PSI) Across Train/Test Splits per Mission", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "distribution_shift_psi.png"), dpi=300, bbox_inches="tight")
    with open(os.path.join(OUT, "distribution_shift.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: distribution_shift.json + distribution_shift_psi.png")


def js_divergence_discrete(p, q):
    p = np.asarray(p) + 1e-9; q = np.asarray(q) + 1e-9
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


if __name__ == "__main__":
    main()

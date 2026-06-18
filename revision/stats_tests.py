"""
Reviewer Point 3: statistical significance analysis.

- McNemar's test (exact binomial) for every pair of deep classifiers per mission,
  using the re-inferred test predictions (revision/results/*_probs.npz).
- Bootstrap 95% confidence intervals (2000 resamples) for accuracy and macro-F1.

statsmodels is NOT required — McNemar is implemented directly with scipy.stats.binom.
"""
import os, json, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

RAW = r"d:\UbtVM-Def\Models\revision\results"
OUT = r"d:\UbtVM-Def\Models\reports\revision"
MODELS = ["cnn", "bilstm", "transformer", "convformer", "hybrid"]
DISP = {"cnn": "CNN", "bilstm": "BiLSTM", "transformer": "Transformer",
        "convformer": "ConvFormer", "hybrid": "Hybrid"}


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact McNemar test. b = A wrong & B right, c = A right & B wrong."""
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    b = int(np.sum(~a_correct & b_correct))
    c = int(np.sum(a_correct & ~b_correct))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "stat": 0.0}
    # exact two-sided binomial p-value with p=0.5
    k = min(b, c)
    p = 2.0 * stats.binom.cdf(k, n, 0.5)
    p = min(1.0, p)
    # chi-square stat with continuity correction (reported alongside)
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return {"b": b, "c": c, "p_value": float(p), "chi2_cc": float(chi2)}


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
    }


def main():
    allres = {}
    for mid in (1, 2, 3):
        preds = {}
        for m in MODELS:
            p = os.path.join(RAW, f"m{mid}_{m}_probs.npz")
            if os.path.exists(p):
                d = np.load(p)
                preds[m] = {"true": d["true"], "pred": d["pred"]}
        if not preds:
            continue
        y_true = preds[list(preds)[0]]["true"]
        # bootstrap CIs
        cis = {DISP[m]: bootstrap_ci(y_true, preds[m]["pred"]) for m in preds}
        # pairwise McNemar
        names = list(preds)
        pairs = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                res = mcnemar_exact(y_true, preds[a]["pred"], preds[b]["pred"])
                res["significant_0.05"] = res["p_value"] < 0.05
                pairs[f"{DISP[a]}_vs_{DISP[b]}"] = res
        allres[f"mission{mid}"] = {"bootstrap_ci": cis, "mcnemar": pairs}
        print(f"\nMission {mid}:")
        for m, c in cis.items():
            lo, hi = c["accuracy_ci95"]
            print(f"  {m:12s} acc={c['accuracy_mean']*100:5.2f}%  CI95=[{lo*100:.2f}, {hi*100:.2f}]  "
                  f"macroF1={c['macro_f1_mean']:.3f}")
        sig = [(k, v["p_value"]) for k, v in pairs.items() if v["significant_0.05"]]
        print(f"  Significant pairs (p<0.05): {len(sig)}/{len(pairs)}")
        for k, p in sorted(sig, key=lambda x: x[1])[:6]:
            print(f"    {k:32s} p={p:.2e}")
    with open(os.path.join(OUT, "statistical_tests.json"), "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\nSaved: {os.path.join(OUT, 'statistical_tests.json')}")


if __name__ == "__main__":
    main()

"""
Reviewer Points 6 + 10: feature importance and preprocessing-component ablation.

The preprocessing produces 5 feature families per base channel:
  base   : Savitzky-Golay smoothed raw signal
  _d1    : first derivative
  _d2    : second derivative
  _rmean : rolling mean
  _rstd  : rolling std

This script:
  (A) Random-Forest impurity importance aggregated by family (which family
      carries the predictive signal?).
  (B) Leave-one-family-out and only-one-family ablation: retrain a RF on each
      feature subset (summary-stat representation) and measure test macro-F1 /
      accuracy delta -> quantifies the NECESSITY of derivative & rolling-stat
      features (Point 10) and the SG/rolling components (Point 6).

Fast: uses RandomForest on summary-stat features, identical split protocol.
"""
import os, json, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

import sys
sys.path.insert(0, r"d:\UbtVM-Def\Models")
import train_all_missions as T

OUT = r"d:\UbtVM-Def\Models\reports\revision"
os.makedirs(OUT, exist_ok=True)
FAMILIES = ["base", "d1", "d2", "rmean", "rstd"]


def family_of(col):
    if col.endswith("_d1"):    return "d1"
    if col.endswith("_d2"):    return "d2"
    if col.endswith("_rmean"): return "rmean"
    if col.endswith("_rstd"):  return "rstd"
    return "base"


def load(mid):
    df = pd.read_csv(os.path.join(T.DATA_DIR, f"mission{mid}_preprocessed.csv"), index_col=0)
    fcols = [c for c in df.columns if c not in ("label", "class_name", "mission_id")]
    X = df[fcols].values.astype(np.float32)
    y_raw = df["label"].values.astype(np.int64)
    uniq = sorted(set(y_raw.tolist()))
    remap = {o: n for n, o in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    Xw, yw = T.make_windows(X, y, T.WINDOW, T.STEP)
    Xtr, ytr, _, _, Xte, yte = T.per_class_chron_split(Xw, yw)
    fam = np.array([family_of(c) for c in fcols])
    return Xtr, ytr, Xte, yte, fam, len(uniq)


def summary(Xw, cols_mask=None):
    if cols_mask is not None:
        Xw = Xw[:, :, cols_mask]
    return np.concatenate([Xw.mean(1), Xw.std(1), Xw.min(1), Xw.max(1), Xw[:, -1, :]], axis=1)


def rf():
    return RandomForestClassifier(n_estimators=200, max_depth=20,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=42)


def evaluate(Xtr, ytr, Xte, yte, mask):
    Ftr, Fte = summary(Xtr, mask), summary(Xte, mask)
    clf = rf(); clf.fit(Ftr, ytr); yp = clf.predict(Fte)
    return {"accuracy": round(accuracy_score(yte, yp) * 100, 2),
            "macro_f1": round(f1_score(yte, yp, average="macro", zero_division=0), 4),
            "balanced_acc": round(balanced_accuracy_score(yte, yp), 4)}


def main():
    allres = {}
    for mid in (1, 2, 3):
        Xtr, ytr, Xte, yte, fam, n_cls = load(mid)
        print(f"\n=== Mission {mid} (n_cls={n_cls}, {len(fam)} features) ===")
        # (A) RF importance by family (full feature set, summary rep)
        # build family mask for summary rep: summary stacks 5 stats x n_feat,
        # so family label repeats 5x in same order
        full = evaluate(Xtr, ytr, Xte, yte, None)
        Ftr = summary(Xtr); clf = rf(); clf.fit(Ftr, ytr)
        imp = clf.feature_importances_
        n_feat = Xtr.shape[2]
        fam_tiled = np.tile(fam, 5)  # 5 summary stats
        fam_imp = {f: float(imp[fam_tiled == f].sum()) for f in FAMILIES}
        tot = sum(fam_imp.values())
        fam_imp = {k: round(v / tot, 4) for k, v in fam_imp.items()}
        print(f"  FULL: acc={full['accuracy']} macroF1={full['macro_f1']}")
        print(f"  RF importance by family: {fam_imp}")
        # (B) only-one-family and leave-one-out ablation
        only, loo = {}, {}
        for f in FAMILIES:
            m_only = (fam == f)
            m_loo  = (fam != f)
            if m_only.sum() > 0:
                only[f] = evaluate(Xtr, ytr, Xte, yte, m_only)
            if m_loo.sum() > 0:
                loo[f] = evaluate(Xtr, ytr, Xte, yte, m_loo)
        for f in FAMILIES:
            d_acc = round(full["accuracy"] - loo[f]["accuracy"], 2)
            print(f"    only[{f:5s}] acc={only[f]['accuracy']:6.2f} mF1={only[f]['macro_f1']:.3f} | "
                  f"drop-{f:5s} acc={loo[f]['accuracy']:6.2f} (delta {d_acc:+.2f})")
        allres[f"mission{mid}"] = {"n_cls": n_cls, "full": full,
                                   "rf_importance_by_family": fam_imp,
                                   "only_family": only, "leave_one_family_out": loo}
    with open(os.path.join(OUT, "feature_importance.json"), "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\nSaved: feature_importance.json")


if __name__ == "__main__":
    main()

"""
Reviewer Point 17 (+ feeds Point 1): classical ML baselines under the IDENTICAL
preprocessing + per-class chronological split protocol as the deep models.

Models: Logistic Regression, Linear SVM, Random Forest, XGBoost, LightGBM.
Each window (WINDOW x n_feat) is flattened to a feature vector. We also try a
compact summary-statistic representation (mean/std/min/max/last per channel)
which is far cheaper and a fairer classical baseline.

Outputs full expanded metrics (acc, weighted/macro F1, balanced acc, MCC,
ROC-AUC, PR-AUC) to reports/revision/classical_baselines.json.
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import lightgbm as lgb

import sys
sys.path.insert(0, r"d:\UbtVM-Def\Models")
import train_all_missions as T

OUT = r"d:\UbtVM-Def\Models\reports\revision"
os.makedirs(OUT, exist_ok=True)


def load_split(mid):
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
    return Xtr, ytr, Xte, yte, len(uniq)


def summary_feats(Xw):
    """Compact per-channel summary stats: mean,std,min,max,last -> 5*n_feat dims."""
    return np.concatenate([Xw.mean(1), Xw.std(1), Xw.min(1), Xw.max(1), Xw[:, -1, :]], axis=1)


def metrics(y_true, y_pred, proba, n_cls):
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        if proba is not None:
            if n_cls == 2:
                m["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
                m["pr_auc"]  = float(average_precision_score(y_true, proba[:, 1]))
            else:
                m["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr",
                                                   average="macro", labels=list(range(n_cls))))
                yb = label_binarize(y_true, classes=list(range(n_cls)))
                m["pr_auc"] = float(average_precision_score(yb, proba, average="macro"))
        else:
            m["roc_auc"] = None; m["pr_auc"] = None
    except Exception as e:
        m["roc_auc"] = None; m["pr_auc"] = None; m["auc_err"] = str(e)
    return m


def get_models(n_cls):
    return {
        "LogisticRegression": LogisticRegression(max_iter=300, class_weight="balanced",
                                                 multi_class="auto", n_jobs=-1),
        "LinearSVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000), cv=3),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=20,
                                               class_weight="balanced_subsample", n_jobs=-1,
                                               random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                     eval_metric="mlogloss", n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.05,
                                       class_weight="balanced", n_jobs=-1, random_state=42,
                                       verbose=-1),
    }


def main():
    allres = {}
    for mid in (1, 2, 3):
        print(f"\n{'='*60}\n  MISSION {mid} — classical baselines\n{'='*60}")
        Xtr, ytr, Xte, yte, n_cls = load_split(mid)
        # compact summary-stat representation (fair + fast)
        Ftr, Fte = summary_feats(Xtr), summary_feats(Xte)
        print(f"  feat dim (summary): {Ftr.shape[1]}  train={len(Ftr)} test={len(Fte)} n_cls={n_cls}")
        mres = {}
        for name, clf in get_models(n_cls).items():
            t0 = time.time()
            try:
                clf.fit(Ftr, ytr)
                y_pred = clf.predict(Fte)
                proba = clf.predict_proba(Fte) if hasattr(clf, "predict_proba") else None
                mm = metrics(yte, y_pred, proba, n_cls)
                mm["train_sec"] = round(time.time() - t0, 1)
                mres[name] = mm
                print(f"  {name:20s} acc={mm['accuracy']*100:5.2f}  macroF1={mm['macro_f1']:.3f}  "
                      f"MCC={mm['mcc']:.3f}  ROC-AUC={mm['roc_auc'] and round(mm['roc_auc'],3)}  "
                      f"({mm['train_sec']}s)")
            except Exception as e:
                print(f"  {name:20s} FAILED: {e}")
                mres[name] = {"error": str(e)}
        allres[f"mission{mid}"] = {"n_cls": n_cls, "feat_dim": int(Ftr.shape[1]), "models": mres}
    with open(os.path.join(OUT, "classical_baselines.json"), "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\nSaved: {os.path.join(OUT, 'classical_baselines.json')}")


if __name__ == "__main__":
    main()

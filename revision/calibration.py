"""
SOTA-matching via post-hoc decision-threshold calibration (Reviewer comments 11, 12).

Diagnosis recap: on Mission 2 the class-weighted deep classifiers retain ROC-AUC ~0.95
(the score ranking is excellent) but collapse in accuracy because the argmax (tau=0.5)
decision threshold is miscalibrated under covariate drift -> they flag ~85% of windows
as anomaly when the true rate is ~14%.

This script recovers accuracy WITHOUT retraining by choosing the operating threshold on
the anomaly score P(anomaly) three ways, for each binary mission:

  baseline   : argmax / tau = 0.5 (what the paper currently reports)
  VAL-BA     : tau* maximising balanced accuracy on the VALIDATION set (supervised,
               no test labels used) -> applied to test
  VAL-F1     : tau* maximising macro-F1 on the validation set -> applied to test
  PRIOR      : flag the top (train base-rate) fraction of test scores. Label-FREE
               unsupervised domain adaptation: assumes the operational anomaly rate
               ~= the training rate (true here, the per-class split preserves priors).

Outputs reports/revision/calibration.json. Uses GPU if torch CUDA is available.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)

import sys
sys.path.insert(0, r"d:\UbtVM-Def\Models")
import train_all_missions as T

DEVICE = T.DEVICE
OUT = r"d:\UbtVM-Def\Models\reports\revision"
os.makedirs(OUT, exist_ok=True)
BIN_MODELS = ["CNN", "BiLSTM", "Transformer", "ConvFormer", "Hybrid"]


def build(name, n_feat, n_cls):
    if name == "CNN":         return T.CNN1D(n_feat, n_cls, T.DROPOUT)
    if name == "BiLSTM":      return T.BiLSTM1D(n_feat, n_cls)
    if name == "Transformer": return T.Transformer1D(n_feat, n_cls)
    if name == "ConvFormer":  return T.ConvFormer1D(n_feat, n_cls)
    if name == "Hybrid":
        return T.HybridModel(T.CNN1D(n_feat, n_cls, T.DROPOUT),
                             T.VAE1D(n_feat, T.WINDOW, T.LATENT_DIM), n_cls)


def load_split(mid):
    import pandas as pd
    df = pd.read_csv(os.path.join(T.DATA_DIR, f"mission{mid}_preprocessed.csv"), index_col=0)
    fcols = [c for c in df.columns if c not in ("label", "class_name", "mission_id")]
    X = df[fcols].values.astype(np.float32)
    y_raw = df["label"].values.astype(np.int64)
    uniq = sorted(set(y_raw.tolist()))
    remap = {o: n for n, o in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    Xw, yw = T.make_windows(X, y, T.WINDOW, T.STEP)
    Xtr, ytr, Xva, yva, Xte, yte = T.per_class_chron_split(Xw, yw)
    return Xtr, ytr, Xva, yva, Xte, yte, X.shape[1], len(uniq)


@torch.no_grad()
def anomaly_score(model, name, X):
    model.eval()
    dl = DataLoader(T.TelDS(X, np.zeros(len(X), np.int64)), T.BATCH, shuffle=False)
    out = []
    for Xb, _ in dl:
        Xb = Xb.to(DEVICE)
        logits = model(Xb)[0] if name == "Hybrid" else model(Xb)
        out.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def metrics(y, p):
    return {"accuracy": round(accuracy_score(y, p) * 100, 2),
            "balanced_acc": round(balanced_accuracy_score(y, p), 4),
            "macro_f1": round(f1_score(y, p, average="macro", zero_division=0), 4),
            "mcc": round(matthews_corrcoef(y, p), 4),
            "precision": round(precision_score(y, p, zero_division=0), 4),
            "recall": round(recall_score(y, p, zero_division=0), 4)}


def best_threshold(y, score, objective):
    grid = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 99)))
    best_t, best_v = 0.5, -1
    for t in grid:
        pred = (score >= t).astype(int)
        v = (balanced_accuracy_score(y, pred) if objective == "ba"
             else f1_score(y, pred, average="macro", zero_division=0))
        if v > best_v:
            best_v, best_t = v, t
    return float(best_t)


def main():
    allres = {}
    for mid in (2, 3):
        Xtr, ytr, Xva, yva, Xte, yte, n_feat, n_cls = load_split(mid)
        if n_cls != 2:
            continue
        prior = float((ytr != 0).mean())  # training anomaly base-rate
        yte_bin = (yte != 0).astype(int)
        yva_bin = (yva != 0).astype(int)
        print(f"\n{'='*66}\n  MISSION {mid}  (test n={len(yte)}, train anomaly rate={prior*100:.1f}%)\n{'='*66}")
        mres = {}
        for name in BIN_MODELS:
            wp = os.path.join(T.MODEL_DIR, f"m{mid}_{name.lower()}.pt")
            if not os.path.exists(wp):
                continue
            model = build(name, n_feat, n_cls).to(DEVICE)
            model.load_state_dict(torch.load(wp, map_location=DEVICE))
            s_va = anomaly_score(model, name, Xva)
            s_te = anomaly_score(model, name, Xte)

            base = metrics(yte_bin, (s_te >= 0.5).astype(int))
            t_ba = best_threshold(yva_bin, s_va, "ba")
            t_f1 = best_threshold(yva_bin, s_va, "f1")
            cal_ba = metrics(yte_bin, (s_te >= t_ba).astype(int))
            cal_f1 = metrics(yte_bin, (s_te >= t_f1).astype(int))
            # PRIOR: flag the top `prior` fraction by score (label-free)
            t_prior = float(np.quantile(s_te, 1 - prior))
            cal_pr = metrics(yte_bin, (s_te >= t_prior).astype(int))

            mres[name] = {"baseline_tau0.5": base,
                          "val_balanced_acc": {"tau": round(t_ba, 4), **cal_ba},
                          "val_macro_f1": {"tau": round(t_f1, 4), **cal_f1},
                          "prior_match": {"tau": round(t_prior, 4), **cal_pr}}
            print(f"  {name:12s} base acc={base['accuracy']:5.2f} -> "
                  f"VAL-BA {cal_ba['accuracy']:5.2f} | VAL-F1 {cal_f1['accuracy']:5.2f} | "
                  f"PRIOR {cal_pr['accuracy']:5.2f}  (mF1 {base['macro_f1']:.3f}->"
                  f"{cal_pr['macro_f1']:.3f})")
        allres[f"mission{mid}"] = {"train_anomaly_rate": round(prior, 4), "models": mres}
    with open(os.path.join(OUT, "calibration.json"), "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\nSaved: calibration.json")


if __name__ == "__main__":
    main()

"""
Reviewer-response evaluation engine.

Re-infers every saved model on the exact (deterministic) per-class chronological
test split and computes the FULL expanded metric suite requested by reviewers:

  Point 4  : Macro-F1, per-class F1, Balanced Accuracy, MCC, ROC-AUC, PR-AUC
  Point 14 : error analysis (most-confused class pairs, per-class error rate)
  Point 15 : per-class precision / recall / F1, confusion matrices
  Point 3  : (consumed by stats_tests.py — this script dumps raw preds+probs)

No retraining: the split is pure index slicing (deterministic), model weights
are loaded from models/. Outputs JSON + markdown tables under reports/revision/.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, r"d:\UbtVM-Def\Models")
import train_all_missions as T

DEVICE = "cpu"
OUT    = r"d:\UbtVM-Def\Models\reports\revision"
RAW    = r"d:\UbtVM-Def\Models\revision\results"
os.makedirs(OUT, exist_ok=True); os.makedirs(RAW, exist_ok=True)

CLF_MODELS = ["CNN", "BiLSTM", "Transformer", "ConvFormer", "Hybrid"]  # multiclass softmax


def build_model(name, n_feat, n_cls):
    if name == "CNN":         return T.CNN1D(n_feat, n_cls, T.DROPOUT)
    if name == "BiLSTM":      return T.BiLSTM1D(n_feat, n_cls)
    if name == "Transformer": return T.Transformer1D(n_feat, n_cls)
    if name == "ConvFormer":  return T.ConvFormer1D(n_feat, n_cls)
    if name == "VAE":         return T.VAE1D(n_feat, T.WINDOW, T.LATENT_DIM)
    if name == "Hybrid":
        cnn = T.CNN1D(n_feat, n_cls, T.DROPOUT)
        vae = T.VAE1D(n_feat, T.WINDOW, T.LATENT_DIM)
        return T.HybridModel(cnn, vae, n_cls)
    raise ValueError(name)


def load_test_split(mid):
    """Reproduce the exact deterministic test set for a mission."""
    import pandas as pd
    csv = os.path.join(T.DATA_DIR, f"mission{mid}_preprocessed.csv")
    df  = pd.read_csv(csv, index_col=0)
    fcols = [c for c in df.columns if c not in ("label", "class_name", "mission_id")]
    X = df[fcols].values.astype(np.float32)
    y_raw = df["label"].values.astype(np.int64)
    uniq = sorted(set(y_raw.tolist()))
    remap = {o: n for n, o in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    names = {remap[o]: T.CLASS_NAMES[o] for o in uniq}
    Xw, yw = T.make_windows(X, y, T.WINDOW, T.STEP)
    Xtr, ytr, Xva, yva, Xte, yte = T.per_class_chron_split(Xw, yw)
    return Xtr, ytr, Xte, yte, len(fcols), len(uniq), names


@torch.no_grad()
def infer_probs(model, name, Xte):
    """Return softmax probabilities (n, n_cls) for classifier models."""
    model.eval()
    dl = DataLoader(T.TelDS(Xte, np.zeros(len(Xte), np.int64)), T.BATCH, shuffle=False)
    out = []
    for Xb, _ in dl:
        Xb = Xb.to(DEVICE)
        if name == "Hybrid":
            logits, *_ = model(Xb)
        else:
            logits = model(Xb)
        out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def vae_scores(model, Xtr, ytr, Xte):
    """Reconstruction-error anomaly score + threshold (mean+2std on normal train)."""
    model.eval()
    norm_x = Xtr[ytr == 0]
    if len(norm_x) == 0: norm_x = Xtr[:100]
    def errs(Xset):
        dl = DataLoader(T.TelDS(Xset, np.zeros(len(Xset), np.int64)), T.BATCH, shuffle=False)
        e = []
        for Xb, _ in dl:
            Xb = Xb.to(DEVICE); r, _, _ = model(Xb)
            e.extend(F.mse_loss(r, Xb, reduction="none").mean(dim=(1, 2)).cpu().tolist())
        return np.array(e)
    en = errs(norm_x); thr = float(en.mean() + 2 * en.std())
    et = errs(Xte)
    return et, thr


def multiclass_metrics(y_true, y_pred, probs, n_cls, names):
    m = {}
    m["accuracy"]          = float(accuracy_score(y_true, y_pred))
    m["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    m["mcc"]               = float(matthews_corrcoef(y_true, y_pred))
    m["weighted_f1"]       = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    m["macro_f1"]          = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    m["weighted_precision"]= float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    m["weighted_recall"]   = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    # ROC-AUC / PR-AUC
    try:
        if n_cls == 2:
            m["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            m["pr_auc"]  = float(average_precision_score(y_true, probs[:, 1]))
        else:
            m["roc_auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr",
                                               average="macro", labels=list(range(n_cls))))
            from sklearn.preprocessing import label_binarize
            yb = label_binarize(y_true, classes=list(range(n_cls)))
            m["pr_auc"] = float(average_precision_score(yb, probs, average="macro"))
    except Exception as e:
        m["roc_auc"] = None; m["pr_auc"] = None; m["auc_error"] = str(e)
    # per-class
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(n_cls)),
                                                 zero_division=0)
    m["per_class"] = {names[c]: {"precision": float(p[c]), "recall": float(r[c]),
                                 "f1": float(f[c]), "support": int(s[c])}
                      for c in range(n_cls)}
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(n_cls))).tolist()
    return m


def error_analysis(y_true, y_pred, n_cls, names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    confused = []
    for i in range(n_cls):
        for j in range(n_cls):
            if i != j and cm[i, j] > 0:
                confused.append({"true": names[i], "pred": names[j], "count": int(cm[i, j]),
                                 "frac_of_true": float(cm[i, j] / max(cm[i].sum(), 1))})
    confused.sort(key=lambda d: d["count"], reverse=True)
    per_class_err = {names[i]: float(1 - cm[i, i] / max(cm[i].sum(), 1)) for i in range(n_cls)}
    return {"top_confusions": confused[:5], "per_class_error_rate": per_class_err}


def main():
    allres = {}
    for mid in (1, 2, 3):
        print(f"\n{'='*60}\n  MISSION {mid}\n{'='*60}")
        Xtr, ytr, Xte, yte, n_feat, n_cls, names = load_test_split(mid)
        print(f"  test windows: {len(Xte)}  n_feat={n_feat}  n_cls={n_cls}  "
              f"dist={ {int(c): int((yte==c).sum()) for c in np.unique(yte)} }")
        mres = {}
        for name in CLF_MODELS:
            wpath = os.path.join(T.MODEL_DIR, f"m{mid}_{name.lower()}.pt")
            if not os.path.exists(wpath):
                print(f"  [skip] {name}: no weights"); continue
            model = build_model(name, n_feat, n_cls).to(DEVICE)
            model.load_state_dict(torch.load(wpath, map_location=DEVICE))
            probs = infer_probs(model, name, Xte)
            y_pred = probs.argmax(1)
            mm = multiclass_metrics(yte, y_pred, probs, n_cls, names)
            mm["error_analysis"] = error_analysis(yte, y_pred, n_cls, names)
            mres[name] = mm
            np.savez(os.path.join(RAW, f"m{mid}_{name.lower()}_probs.npz"),
                     true=yte, pred=y_pred, probs=probs)
            print(f"  {name:12s} acc={mm['accuracy']*100:5.2f}  macroF1={mm['macro_f1']:.3f}  "
                  f"balAcc={mm['balanced_accuracy']:.3f}  MCC={mm['mcc']:.3f}  "
                  f"ROC-AUC={mm['roc_auc'] if mm['roc_auc'] is None else round(mm['roc_auc'],3)}  "
                  f"PR-AUC={mm['pr_auc'] if mm['pr_auc'] is None else round(mm['pr_auc'],3)}")
        # VAE — binary reconstruction track
        vpath = os.path.join(T.MODEL_DIR, f"m{mid}_vae.pt")
        if os.path.exists(vpath):
            vae = build_model("VAE", n_feat, n_cls).to(DEVICE)
            vae.load_state_dict(torch.load(vpath, map_location=DEVICE))
            et, thr = vae_scores(vae, Xtr, ytr, Xte)
            vt = (yte != 0).astype(int); vp = (et > thr).astype(int)
            vm = {
                "task": "binary (normal vs anomaly, reconstruction error)",
                "accuracy": float(accuracy_score(vt, vp)),
                "balanced_accuracy": float(balanced_accuracy_score(vt, vp)),
                "mcc": float(matthews_corrcoef(vt, vp)),
                "f1": float(f1_score(vt, vp, zero_division=0)),
                "precision": float(precision_score(vt, vp, zero_division=0)),
                "recall": float(recall_score(vt, vp, zero_division=0)),
                "roc_auc": float(roc_auc_score(vt, et)) if len(set(vt)) > 1 else None,
                "pr_auc": float(average_precision_score(vt, et)) if len(set(vt)) > 1 else None,
                "threshold": thr,
            }
            mres["VAE"] = vm
            np.savez(os.path.join(RAW, f"m{mid}_vae_scores.npz"), true=vt, score=et, thr=thr)
            print(f"  {'VAE':12s} acc={vm['accuracy']*100:5.2f}  ROC-AUC={round(vm['roc_auc'],3)}  "
                  f"PR-AUC={round(vm['pr_auc'],3)}  (binary recon)")
        allres[f"mission{mid}"] = {"n_feat": n_feat, "n_cls": n_cls,
                                   "test_windows": int(len(Xte)),
                                   "class_names": names, "models": mres}
    with open(os.path.join(OUT, "expanded_metrics.json"), "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\nSaved: {os.path.join(OUT, 'expanded_metrics.json')}")
    return allres


if __name__ == "__main__":
    main()

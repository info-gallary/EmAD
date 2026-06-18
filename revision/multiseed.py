"""
Reviewer Point 2: multi-seed evaluation with mean +/- std.

Retrains the four deep classifiers (CNN, BiLSTM, Transformer, ConvFormer) on each
mission with 3 independent seeds and evaluates each on the SAME deterministic
per-class chronological test split. Reports mean +/- std for accuracy, weighted-F1,
macro-F1, balanced accuracy and MCC.

Weights are written to models/multiseed/ so the canonical single-seed weights
(used by the other revision scripts) are NOT overwritten.

This is the heaviest revision job — intended to run in the background.
"""
import os, json, time, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             matthews_corrcoef)

import sys
sys.path.insert(0, r"d:\UbtVM-Def\Models")
import train_all_missions as T

DEVICE = T.DEVICE
SEEDS = [42, 3, 7]
OUT = r"d:\UbtVM-Def\Models\reports\revision"
WDIR = r"d:\UbtVM-Def\Models\models\multiseed"
os.makedirs(OUT, exist_ok=True); os.makedirs(WDIR, exist_ok=True)
MAX_EP = 120  # early stopping cuts this short


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
    return T.per_class_chron_split(Xw, yw) + (X.shape[1], len(uniq))


def evaluate(model, name, tstl, yte):
    model.eval(); ps = []
    with torch.no_grad():
        for Xb, _ in tstl:
            logits = model(Xb.to(DEVICE))
            if name == "Hybrid": logits = logits[0]
            ps.extend(logits.argmax(1).cpu().tolist())
    ps = np.array(ps)
    return {
        "accuracy": accuracy_score(yte, ps) * 100,
        "weighted_f1": f1_score(yte, ps, average="weighted", zero_division=0),
        "macro_f1": f1_score(yte, ps, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(yte, ps),
        "mcc": matthews_corrcoef(yte, ps),
    }


def train_one(name, mid, seed, data):
    Xtr, ytr, Xva, yva, Xte, yte, n_feat, n_cls = data
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    cw = T.class_weights(ytr, n_cls, DEVICE)
    trnl = DataLoader(T.TelDS(Xtr, ytr), T.BATCH, shuffle=True)
    vall = DataLoader(T.TelDS(Xva, yva), T.BATCH, shuffle=False)
    tstl = DataLoader(T.TelDS(Xte, yte), T.BATCH, shuffle=False)

    if name == "CNN":
        model = T.CNN1D(n_feat, n_cls, T.DROPOUT).to(DEVICE); loader = trnl; focal = None
    elif name == "BiLSTM":
        model = T.BiLSTM1D(n_feat, n_cls, dr=T.DROPOUT).to(DEVICE); loader = trnl; focal = None
    elif name == "Transformer":
        model = T.Transformer1D(n_feat, n_cls, dr=T.DROPOUT).to(DEVICE); loader = trnl; focal = None
    elif name == "ConvFormer":
        model = T.ConvFormer1D(n_feat, n_cls, dr=T.DROPOUT).to(DEVICE)
        loader = T.make_balanced_loader(Xtr, ytr, T.BATCH)
        focal = T.FocalLoss(gamma=2.0, weight=cw)

    opt = optim.AdamW(model.parameters(), lr=T.LR, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EP, eta_min=1e-5)
    best_va, best_w, no_imp = 0.0, None, 0
    for ep in range(1, MAX_EP + 1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
            out = model(Xb)
            loss = focal(out, yb) if focal is not None else \
                F.cross_entropy(out, yb, weight=cw, label_smoothing=0.05)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            va = vn = 0
            for Xb, yb in vall:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                va += (model(Xb).argmax(1) == yb).sum().item(); vn += len(yb)
        va_acc = va / vn
        if va_acc > best_va:
            best_va = va_acc; best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
        if no_imp >= T.PATIENCE:
            break
    model.load_state_dict(best_w)
    torch.save(best_w, os.path.join(WDIR, f"m{mid}_{name.lower()}_s{seed}.pt"))
    return evaluate(model, name, tstl, yte), ep


def agg(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "values": [round(float(v), 4) for v in vals]}


def main():
    models = ["CNN", "BiLSTM", "Transformer", "ConvFormer"]
    allres = {}
    t_start = time.time()
    for mid in (1, 2, 3):
        data = load_split(mid)
        print(f"\n{'='*64}\n  MISSION {mid}  (n_feat={data[6]}, n_cls={data[7]})\n{'='*64}")
        mres = {}
        for name in models:
            runs = []
            for seed in SEEDS:
                t0 = time.time()
                m, ep = train_one(name, mid, seed, data)
                runs.append(m)
                print(f"  {name:12s} seed={seed:3d}  acc={m['accuracy']:5.2f}  "
                      f"wF1={m['weighted_f1']:.3f}  mF1={m['macro_f1']:.3f}  "
                      f"MCC={m['mcc']:.3f}  (ep={ep}, {time.time()-t0:.0f}s)")
            mres[name] = {k: agg([r[k] for r in runs]) for k in runs[0]}
            a = mres[name]["accuracy"]
            print(f"  -> {name:12s} acc = {a['mean']:.2f} +/- {a['std']:.2f}  "
                  f"(seeds {SEEDS})")
        allres[f"mission{mid}"] = mres
        # checkpoint after each mission
        with open(os.path.join(OUT, "multiseed_results.json"), "w") as f:
            json.dump({"seeds": SEEDS, "results": allres}, f, indent=2)
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    print(f"Saved: multiseed_results.json")


if __name__ == "__main__":
    main()

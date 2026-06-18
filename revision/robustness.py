"""
Reviewer Point 16: robustness to sensor noise and missing data.

Perturbs the held-out test windows and re-infers with the saved weights
(no retraining). Two perturbation families, each at several severities:

  - Additive Gaussian noise at signal-to-noise ratios (features are z-scored,
    so noise std sigma directly controls SNR): sigma in {0.05, 0.1, 0.2, 0.5}
  - Random missing values imputed by forward-fill / zero: frac in {0.05,0.1,0.2,0.4}

Reports accuracy + macro-F1 vs perturbation severity for each model/mission,
plus an area-under-degradation summary (mean retained accuracy).
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.insert(0, r"d:\UbtVM-Def\Models")
import train_all_missions as T

DEVICE = "cpu"
OUT = r"d:\UbtVM-Def\Models\reports\revision"
os.makedirs(OUT, exist_ok=True)
CLF = ["CNN", "BiLSTM", "Transformer", "ConvFormer", "Hybrid"]
rng = np.random.default_rng(0)


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
    csv = os.path.join(T.DATA_DIR, f"mission{mid}_preprocessed.csv")
    df = pd.read_csv(csv, index_col=0)
    fcols = [c for c in df.columns if c not in ("label", "class_name", "mission_id")]
    X = df[fcols].values.astype(np.float32)
    y_raw = df["label"].values.astype(np.int64)
    uniq = sorted(set(y_raw.tolist()))
    remap = {o: n for n, o in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=np.int64)
    Xw, yw = T.make_windows(X, y, T.WINDOW, T.STEP)
    _, _, _, _, Xte, yte = T.per_class_chron_split(Xw, yw)
    return Xte, yte, len(fcols), len(uniq)


@torch.no_grad()
def infer(model, name, Xte):
    model.eval()
    dl = DataLoader(T.TelDS(Xte, np.zeros(len(Xte), np.int64)), T.BATCH, shuffle=False)
    out = []
    for Xb, _ in dl:
        logits = model(Xb)[0] if name == "Hybrid" else model(Xb)
        out.append(torch.softmax(logits, 1).argmax(1).cpu().numpy())
    return np.concatenate(out)


def add_noise(Xte, sigma):
    return Xte + rng.normal(0, sigma, Xte.shape).astype(np.float32)


def add_missing(Xte, frac):
    """Randomly drop values; impute by vectorised forward-fill along time, then
    mean-impute (z-scored features => 0) any leading gaps. Fully vectorised."""
    X = Xte.copy()
    mask = rng.random(X.shape) < frac
    X[mask] = np.nan
    # vectorised forward-fill along the time axis (axis=1) for all windows/channels
    n, t, c = X.shape
    valid = ~np.isnan(X)
    # index of last valid timestep at or before each position
    idx = np.where(valid, np.arange(t)[None, :, None], 0)
    idx = np.maximum.accumulate(idx, axis=1)
    ff = np.take_along_axis(X, idx, axis=1)
    # leading NaNs (no prior valid value) -> 0 (mean of z-scored signal)
    return np.nan_to_num(ff, nan=0.0).astype(np.float32)


def main():
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.5]
    fracs  = [0.0, 0.05, 0.1, 0.2, 0.4]
    allres = {}
    for mid in (1, 2, 3):
        Xte, yte, n_feat, n_cls = load_split(mid)
        print(f"\n=== Mission {mid} (n={len(Xte)}, n_cls={n_cls}) ===")
        mres = {}
        for name in CLF:
            wp = os.path.join(T.MODEL_DIR, f"m{mid}_{name.lower()}.pt")
            if not os.path.exists(wp): continue
            model = build(name, n_feat, n_cls).to(DEVICE)
            model.load_state_dict(torch.load(wp, map_location=DEVICE))
            noise_acc, miss_acc = [], []
            for s in sigmas:
                yp = infer(model, name, add_noise(Xte, s) if s > 0 else Xte)
                noise_acc.append(round(accuracy_score(yte, yp) * 100, 2))
            for fr in fracs:
                yp = infer(model, name, add_missing(Xte, fr) if fr > 0 else Xte)
                miss_acc.append(round(accuracy_score(yte, yp) * 100, 2))
            clean = noise_acc[0]
            # retained accuracy = mean over perturbed / clean
            ret_noise = round(np.mean(noise_acc[1:]) / max(clean, 1e-6) * 100, 1)
            ret_miss  = round(np.mean(miss_acc[1:]) / max(clean, 1e-6) * 100, 1)
            mres[name] = {"clean_acc": clean,
                          "noise_sigma": sigmas, "noise_acc": noise_acc,
                          "missing_frac": fracs, "missing_acc": miss_acc,
                          "retained_pct_noise": ret_noise, "retained_pct_missing": ret_miss}
            print(f"  {name:12s} clean={clean:5.2f} | noise@0.2={noise_acc[3]:5.2f} "
                  f"miss@0.2={miss_acc[3]:5.2f} | retained noise={ret_noise:5.1f}% miss={ret_miss:5.1f}%")
        allres[f"mission{mid}"] = mres
    with open(os.path.join(OUT, "robustness.json"), "w") as f:
        json.dump(allres, f, indent=2)
    print(f"\nSaved: robustness.json")


if __name__ == "__main__":
    main()

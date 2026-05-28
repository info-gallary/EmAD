"""
Train CNN, VAE, and Hybrid Meta-Learner on each of the 3 ESA missions.

Inputs  (from preprocess_all_missions.py)
---------
  data/mission1_preprocessed.csv
  data/mission2_preprocessed.csv
  data/mission3_preprocessed.csv

Outputs
-------
  models/mN_cnn.pt          -- per-mission CNN weights
  models/mN_vae.pt          -- per-mission VAE weights
  models/mN_hybrid.pt       -- per-mission Hybrid weights
  reports/missions/mN/      -- per-mission plots + metrics
  reports/missions/cross_mission_comparison.png
  reports/missions/all_missions_summary.txt
"""

import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score,
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ---- paths ------------------------------------------------------------------
DATA_DIR   = r"d:\UbtVM-Def\Models\data"
MODEL_DIR  = r"d:\UbtVM-Def\Models\models"
REPORT_DIR = r"d:\UbtVM-Def\Models\reports\missions"

# ---- hyper-params -----------------------------------------------------------
WINDOW     = 50
STEP       = 2
BATCH      = 256
CNN_EP     = 120   # upper bound — early stopping will cut much earlier
VAE_EP     = 60
PHASE1_EP  = 60
PHASE2_EP  = 80
PATIENCE   = 12    # epochs with no val improvement before stopping
SEEDS      = [42]   # single seed run
LR         = 1e-3
LR2        = 5e-5
DROPOUT    = 0.4
LATENT_DIM = 64
W_CLS      = 1.0
W_REC      = 0.3
W_KL       = 0.05
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "Normal",          1: "Comm. Anomaly",   2: "Power Anomaly",
    3: "Thermal Anomaly", 4: "Software Anomaly", 5: "Rare-Event",
    6: "Comm-Gap",        7: "Unknown Anomaly",
}
PALETTE = ["#2196F3","#FF5722","#9C27B0","#F44336","#FF9800","#4CAF50","#00BCD4","#607D8B"]


# ---- publication style ------------------------------------------------------

def set_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "axes.titleweight": "bold", "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9,
        "legend.framealpha": 0.9, "figure.facecolor": "white",
        "axes.facecolor": "#f9f9f9", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.35, "grid.linestyle": "--",
        "lines.linewidth": 2.0,
    })


# ---- dataset ----------------------------------------------------------------

def make_windows(X, y, win, step):
    Xw, yw = [], []
    for i in range(0, len(X) - win + 1, step):
        seg = y[i:i + win]; v, c = np.unique(seg, return_counts=True)
        Xw.append(X[i:i + win]); yw.append(v[c.argmax()])
    return np.array(Xw, np.float32), np.array(yw, np.int64)


def per_class_chron_split(Xw, yw, test_r=0.15, val_r=0.15):
    """Stratified chronological split: each class donates its latest windows to
    test/val, preserving temporal order and guaranteeing all classes appear in
    every split — eliminates trivial single-class test sets."""
    tr_idx, va_idx, te_idx = [], [], []
    for cls in np.unique(yw):
        idx = np.where(yw == cls)[0]
        n = len(idx)
        n_te = max(1, int(n * test_r))
        n_va = max(1, int(n * val_r))
        te_idx.extend(idx[-n_te:].tolist())
        va_idx.extend(idx[-(n_te + n_va):-n_te].tolist())
        tr_idx.extend(idx[:-(n_te + n_va)].tolist())
    return (Xw[sorted(tr_idx)], yw[sorted(tr_idx)],
            Xw[sorted(va_idx)], yw[sorted(va_idx)],
            Xw[sorted(te_idx)], yw[sorted(te_idx)])


class TelDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ---- model definitions ------------------------------------------------------

class Res1D(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        p = k // 2
        self.n = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=p, bias=False), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, k, padding=p, bias=False), nn.BatchNorm1d(ch))
        self.a = nn.GELU()
    def forward(self, x): return self.a(x + self.n(x))


class CNN1D(nn.Module):
    def __init__(self, nf, nc, dr=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(nf, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.GELU())
        self.s1 = nn.Sequential(Res1D(64), Res1D(64))
        self.d1 = nn.Sequential(nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(128), nn.GELU())
        self.s2 = nn.Sequential(Res1D(128), Res1D(128))
        self.d2 = nn.Sequential(nn.Conv1d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(256), nn.GELU())
        self.s3 = nn.Sequential(Res1D(256), Res1D(256))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.GELU(), nn.Dropout(dr), nn.Linear(128, nc))
    def forward(self, x):
        x = self.stem(x); x = self.d1(self.s1(x)); x = self.d2(self.s2(x)); x = self.s3(x)
        return self.head(self.pool(x))


class BiLSTM1D(nn.Module):
    def __init__(self, nf, nc, hidden=128, nlayers=2, dr=0.4):
        super().__init__()
        self.lstm = nn.LSTM(nf, hidden, nlayers, batch_first=True,
                            dropout=dr if nlayers > 1 else 0.0, bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Dropout(dr), nn.Linear(hidden * 2, nc))
    def forward(self, x):
        out, _ = self.lstm(x.permute(0, 2, 1))
        return self.head(out[:, -1, :])


class Transformer1D(nn.Module):
    def __init__(self, nf, nc, d_model=128, nhead=4, nlayers=2, dr=0.4):
        super().__init__()
        self.proj = nn.Linear(nf, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                               dropout=dr, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dr), nn.Linear(d_model, nc))
    def forward(self, x):
        x = self.proj(x.permute(0, 2, 1))
        x = self.enc(x)
        return self.head(x.mean(dim=1))


class ConvFormer1D(nn.Module):
    """
    Novel CNN-Transformer hybrid (proposed method).
    CNN stem creates rich local patch tokens → fixes minority class collapse.
    Transformer encoder captures global temporal context → handles distribution shift.
    Learnable positional embedding + PreNorm layers for training stability.
    ~1.2 M parameters — lightweight, single-pass inference.
    """
    def __init__(self, nf, nc, d_model=128, nhead=4, nlayers=2, dr=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(nf, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, d_model, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(d_model), nn.GELU())          # → (B, d_model, 25)
        self.pos_emb = nn.Parameter(torch.zeros(1, 25, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2,
                                         dropout=dr, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dr), nn.Linear(d_model, nc))
    def forward(self, x):
        x = self.stem(x).permute(0, 2, 1) + self.pos_emb   # (B,25,d_model)
        return self.head(self.encoder(x).mean(1))


class VAE1D(nn.Module):
    def __init__(self, nf, win, ld=64):
        super().__init__()
        self.win = win; PT = 8
        self.enc = nn.Sequential(
            nn.Conv1d(nf, 128, 7, padding=3, bias=False), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 5, padding=2, bias=False), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(PT))
        fl = 256 * PT
        self.mu_l  = nn.Linear(fl, ld)
        self.lv_l  = nn.Linear(fl, ld)
        self.dec_fc = nn.Linear(ld, fl)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (256, PT)),
            nn.ConvTranspose1d(256, 128, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.ConvTranspose1d(128,  64, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm1d(64),  nn.GELU(),
            nn.ConvTranspose1d( 64,  nf, 5, padding=2), nn.Sigmoid())
    def encode(self, x): h = self.enc(x).flatten(1); return self.mu_l(h), self.lv_l(h)
    def reparam(self, m, l): return m + (0.5 * l).exp() * torch.randn_like(m)
    def decode(self, z): return F.interpolate(self.dec(self.dec_fc(z)), size=self.win, mode="linear", align_corners=False)
    def forward(self, x):
        m, l = self.encode(x); return self.decode(self.reparam(m, l)), m, l


class HybridModel(nn.Module):
    def __init__(self, cnn: CNN1D, vae: VAE1D, n_cls: int):
        super().__init__()
        self.cnn = cnn; self.vae = vae
        self.meta = nn.Sequential(
            nn.Linear(320, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_cls))
    def cnn_feat(self, x):
        x = self.cnn.stem(x)
        x = self.cnn.d1(self.cnn.s1(x))
        x = self.cnn.d2(self.cnn.s2(x))
        x = self.cnn.s3(x)
        return self.cnn.pool(x).squeeze(-1)
    def forward(self, x):
        feat = self.cnn_feat(x)
        recon, mu, lv = self.vae(x)
        return self.meta(torch.cat([feat, mu], dim=1)), recon, mu, lv
    def freeze_backbones(self):
        for p in self.cnn.parameters(): p.requires_grad_(False)
        for p in self.vae.parameters(): p.requires_grad_(False)
    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad_(True)


# ---- losses -----------------------------------------------------------------

def class_weights(ytr, n_cls, device, max_w=3.0):
    ci, cn = np.unique(ytr, return_counts=True)
    wts = np.ones(n_cls, np.float32)
    for c, n in zip(ci, cn): wts[c] = min(len(ytr) / (len(ci) * n), max_w)
    return torch.tensor(wts).to(device)


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017): down-weights easy majority-class examples
    so the model focuses training on hard minority classes (e.g. Thermal Anomaly)."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma; self.weight = weight
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction="none", label_smoothing=0.05)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def make_balanced_loader(X, y, batch_size):
    """WeightedRandomSampler: every mini-batch is class-balanced regardless of
    natural distribution — critical for severe imbalance (e.g. M2 at 93% Normal)."""
    cls, cnts = np.unique(y, return_counts=True)
    w = np.zeros(len(y), dtype=np.float32)
    for c, n in zip(cls, cnts):
        w[y == c] = 1.0 / n
    sampler = WeightedRandomSampler(torch.from_numpy(w), len(w), replacement=True)
    return DataLoader(TelDS(X, y), batch_size, sampler=sampler, num_workers=0)


def joint_loss(logits, y, recon, x, mu, lv, cw):
    cls_l = F.cross_entropy(logits, y, weight=cw, label_smoothing=0.05)
    rec_l = F.mse_loss(recon, x)
    kl_l  = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return W_CLS*cls_l + W_REC*rec_l + W_KL*kl_l, cls_l.item(), rec_l.item(), kl_l.item()


# ---- training loops ---------------------------------------------------------

def train_cnn_epoch(model, dl, cw, opt):
    model.train(); tl = tc = tn = 0
    for X, y in dl:
        X, y = X.to(DEVICE), y.to(DEVICE); opt.zero_grad()
        out = model(X)
        loss = F.cross_entropy(out, y, weight=cw, label_smoothing=0.05)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item()*len(y); tc += (out.argmax(1)==y).sum().item(); tn += len(y)
    return tl/tn, tc/tn


def train_vae_epoch(model, dl, opt):
    model.train(); tl = tn = 0
    for X, _ in dl:
        X = X.to(DEVICE); opt.zero_grad()
        r, m, l = model(X)
        rec = F.mse_loss(r, X)
        kl  = -0.5 * torch.mean(1 + l - m.pow(2) - l.exp())
        loss = rec + 0.1*kl
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item()*len(X); tn += len(X)
    return tl/tn


def train_hybrid_epoch(model, dl, cw, opt, phase2=False):
    model.train(); tl = tc = tn = 0
    for X, y in dl:
        X, y = X.to(DEVICE), y.to(DEVICE); opt.zero_grad()
        logits, recon, mu, lv = model(X)
        if phase2:
            loss, *_ = joint_loss(logits, y, recon, X, mu, lv, cw)
        else:
            loss = F.cross_entropy(logits, y, weight=cw, label_smoothing=0.05)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item()*len(y); tc += (logits.argmax(1)==y).sum().item(); tn += len(y)
    return tl/tn, tc/tn


@torch.no_grad()
def eval_cnn(model, dl):
    model.eval(); ps, ls, prbs = [], [], []
    for X, y in dl:
        X, y = X.to(DEVICE), y.to(DEVICE); out = model(X)
        ps.extend(out.argmax(1).cpu().tolist())
        ls.extend(y.cpu().tolist())
        prbs.append(torch.softmax(out,1).cpu().numpy())
    return np.array(ps), np.array(ls), np.vstack(prbs)


@torch.no_grad()
def eval_hybrid(model, dl):
    model.eval(); ps, ls, prbs, mus = [], [], [], []
    for X, y in dl:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits, _, mu, _ = model(X)
        ps.extend(logits.argmax(1).cpu().tolist())
        ls.extend(y.cpu().tolist())
        prbs.append(torch.softmax(logits,1).cpu().numpy())
        mus.append(mu.cpu().numpy())
    return np.array(ps), np.array(ls), np.vstack(prbs), np.vstack(mus)


# ---- plots ------------------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, names, title, out):
    present = sorted(set(y_true) | set(y_pred))
    labs = [names[i] for i in present]
    cm  = confusion_matrix(y_true, y_pred, labels=present)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, fmt, t in zip(axes, [cm, cmn], ["d", ".2%"], ["Counts", "Row-Normalised"]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labs, yticklabels=labs, ax=ax,
                    linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.8})
        ax.set(xlabel="Predicted", ylabel="True", title=f"{title} ({t})")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()


def save_roc(y_true, probs, names, n_cls, title, out):
    present = sorted(set(y_true))
    yb = label_binarize(y_true, classes=list(range(n_cls)))
    if n_cls == 2:  # label_binarize returns (n,1) for binary; expand to (n,2)
        yb = np.hstack([1 - yb, yb])
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls in present:
        if cls >= yb.shape[1] or yb[:, cls].sum() == 0: continue
        fpr, tpr, _ = roc_curve(yb[:, cls], probs[:, cls])
        ax.plot(fpr, tpr, color=PALETTE[cls % len(PALETTE)], lw=2,
                label=f"{names[cls]}  (AUC={auc(fpr,tpr):.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title=f"{title} ROC"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()


def save_tsne(mus, y_true, names, title, out):
    print("    t-SNE ...")
    z2 = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=800, init="pca").fit_transform(mus)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in sorted(set(y_true)):
        idx = y_true == cls
        ax.scatter(z2[idx,0], z2[idx,1], c=PALETTE[cls % len(PALETTE)], s=10,
                   alpha=0.6, label=names[cls])
    ax.set(title=f"{title} VAE Latent Space (t-SNE)"); ax.legend(markerscale=2.5)
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()


def save_training_curve(train_losses, val_accs, title, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(train_losses, color="#1565C0"); a1.set(title=f"{title} Loss", xlabel="Epoch", ylabel="Loss")
    a2.plot([v*100 for v in val_accs], color="#388E3C"); a2.set(title=f"{title} Val Accuracy", xlabel="Epoch", ylabel="Acc (%)")
    plt.suptitle(title, fontweight="bold"); plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()


# ---- per-mission pipeline ---------------------------------------------------

def run_mission(mid):
    csv_path = os.path.join(DATA_DIR, f"mission{mid}_preprocessed.csv")
    rdir = os.path.join(REPORT_DIR, f"m{mid}")
    os.makedirs(rdir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  MISSION {mid}  -->  {csv_path}")
    print(f"{'='*62}")

    df     = pd.read_csv(csv_path, index_col=0)
    fcols  = [c for c in df.columns if c not in ("label","class_name","mission_id")]
    X      = df[fcols].values.astype(np.float32)
    y_raw  = df["label"].values.astype(np.int64)

    # remap labels to contiguous 0..K-1
    uniq_cls = sorted(set(y_raw.tolist()))
    remap    = {orig: new for new, orig in enumerate(uniq_cls)}
    y        = np.array([remap[v] for v in y_raw], dtype=np.int64)
    n_cls    = len(uniq_cls)
    n_feat   = X.shape[1]

    local_names = {new: CLASS_NAMES[orig] for orig, new in remap.items()}
    print(f"  Samples:{len(X):,}  Features:{n_feat}  Classes:{uniq_cls}  (remapped 0..{n_cls-1})")

    Xw, yw = make_windows(X, y, WINDOW, STEP)
    # per-class chronological split — each class's latest windows go to test
    Xtr, ytr, Xva, yva, Xte, yte = per_class_chron_split(Xw, yw)
    print(f"  Windows  train:{len(Xtr):,}  val:{len(Xva):,}  test:{len(Xte):,}")
    print(f"  Test class dist: { {c: int((yte==c).sum()) for c in np.unique(yte)} }")

    cw   = class_weights(ytr, n_cls, DEVICE)
    trnl = DataLoader(TelDS(Xtr, ytr), BATCH, shuffle=True, num_workers=0)
    vall = DataLoader(TelDS(Xva, yva), BATCH, shuffle=False, num_workers=0)
    tstl = DataLoader(TelDS(Xte, yte), BATCH, shuffle=False, num_workers=0)

    results = {}

    # ---- CNN ----------------------------------------------------------------
    print(f"\n  [CNN] Training {CNN_EP} epochs ...")
    cnn = CNN1D(n_feat, n_cls, DROPOUT).to(DEVICE)
    opt = optim.AdamW(cnn.parameters(), lr=LR, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CNN_EP, eta_min=1e-5)
    best_va = 0.0; best_w = None; tr_losses = []; va_accs = []; no_improve = 0

    for ep in range(1, CNN_EP+1):
        tl, ta = train_cnn_epoch(cnn, trnl, cw, opt)

        cnn.eval()
        with torch.no_grad():
            va = 0; vn = 0
            for X_, y_ in vall:
                X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
                va += (cnn(X_).argmax(1)==y_).sum().item(); vn += len(y_)
        va_acc = va / vn
        sch.step()
        tr_losses.append(tl); va_accs.append(va_acc)
        if va_acc > best_va:
            best_va = va_acc; best_w = {k:v.cpu().clone() for k,v in cnn.state_dict().items()}; no_improve = 0
        else:
            no_improve += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:02d}/{CNN_EP}  train {ta*100:.2f}%  val {va_acc*100:.2f}%")
        if no_improve >= PATIENCE:
            print(f"    Early stop at ep {ep} (no val improvement for {PATIENCE} epochs)")
            break

    cnn.load_state_dict(best_w)
    cnn_pt = os.path.join(MODEL_DIR, f"m{mid}_cnn.pt")
    torch.save(best_w, cnn_pt)
    cnn_pred, cnn_true, cnn_probs = eval_cnn(cnn, tstl)
    cnn_acc  = accuracy_score(cnn_true, cnn_pred)
    cnn_f1   = f1_score(cnn_true, cnn_pred, average="weighted", zero_division=0)
    cnn_prec = precision_score(cnn_true, cnn_pred, average="weighted", zero_division=0)
    cnn_rec  = recall_score(cnn_true, cnn_pred, average="weighted", zero_division=0)
    results["CNN"] = (cnn_acc, cnn_f1, cnn_prec, cnn_rec)
    print(f"    Test Acc {cnn_acc*100:.2f}%  F1 {cnn_f1:.4f}")

    save_training_curve(tr_losses, va_accs, f"Mission {mid} CNN", os.path.join(rdir, "cnn_training_curve.png"))
    save_confusion_matrix(cnn_true, cnn_pred, local_names, f"M{mid} CNN", os.path.join(rdir, "cnn_confusion_matrix.png"))
    save_roc(cnn_true, cnn_probs, local_names, n_cls, f"M{mid} CNN", os.path.join(rdir, "cnn_roc.png"))

    # ---- BiLSTM baseline ----------------------------------------------------
    def train_baseline(model, tag):
        opt_b = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sch_b = optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=CNN_EP, eta_min=1e-5)
        best_bva = 0.0; best_bw = None; no_imp = 0
        tr_hist = []; va_hist = []
        print(f"\n  [{tag}] Training (max {CNN_EP} ep, patience={PATIENCE}) ...")
        for ep in range(1, CNN_EP + 1):
            tl, ta = train_cnn_epoch(model, trnl, cw, opt_b); sch_b.step()
            model.eval()
            with torch.no_grad():
                va = vn = 0
                for X_, y_ in vall:
                    X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
                    va += (model(X_).argmax(1)==y_).sum().item(); vn += len(y_)
            va_acc = va / vn
            tr_hist.append(tl); va_hist.append(va_acc)
            if va_acc > best_bva:
                best_bva = va_acc; best_bw = {k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp = 0
            else:
                no_imp += 1
            if ep % 10 == 0 or ep == 1:
                print(f"    Ep {ep:02d}  train {ta*100:.2f}%  val {va_acc*100:.2f}%")
            if no_imp >= PATIENCE:
                print(f"    Early stop at ep {ep}"); break
        model.load_state_dict(best_bw)
        # save weights for post-hoc analysis
        torch.save(best_bw, os.path.join(MODEL_DIR, f"m{mid}_{tag.lower()}.pt"))
        bp, bt, bpr = eval_cnn(model, tstl)
        b_acc  = accuracy_score(bt, bp)
        b_f1   = f1_score(bt, bp, average="weighted", zero_division=0)
        b_prec = precision_score(bt, bp, average="weighted", zero_division=0)
        b_rec  = recall_score(bt, bp, average="weighted", zero_division=0)
        print(f"    Test Acc {b_acc*100:.2f}%  F1 {b_f1:.4f}")
        save_confusion_matrix(bt, bp, local_names, f"M{mid} {tag}", os.path.join(rdir, f"{tag.lower()}_confusion_matrix.png"))
        save_roc(bt, bpr, local_names, n_cls, f"M{mid} {tag}", os.path.join(rdir, f"{tag.lower()}_roc.png"))
        save_training_curve(tr_hist, va_hist, f"Mission {mid} {tag}", os.path.join(rdir, f"{tag.lower()}_training_curve.png"))
        return (b_acc, b_f1, b_prec, b_rec), bp, bt, bpr

    bilstm = BiLSTM1D(n_feat, n_cls, dr=DROPOUT).to(DEVICE)
    results["BiLSTM"], lstm_pred, lstm_true, lstm_probs = train_baseline(bilstm, "BiLSTM")

    transformer = Transformer1D(n_feat, n_cls, dr=DROPOUT).to(DEVICE)
    results["Transformer"], tf_pred, tf_true, tf_probs = train_baseline(transformer, "Transformer")

    # ---- ConvFormer (proposed) — CNN patches + Transformer + Focal Loss ------
    print(f"\n  [ConvFormer] Training (max {CNN_EP} ep, patience={PATIENCE}) ...")
    convformer = ConvFormer1D(n_feat, n_cls, dr=DROPOUT).to(DEVICE)
    focal_loss = FocalLoss(gamma=2.0, weight=cw)
    bal_trnl   = make_balanced_loader(Xtr, ytr, BATCH)
    cf_opt  = optim.AdamW(convformer.parameters(), lr=LR, weight_decay=1e-4)
    cf_sch  = optim.lr_scheduler.CosineAnnealingLR(cf_opt, T_max=CNN_EP, eta_min=1e-5)
    cf_best_va = 0.0; cf_best_w = None; cf_no_imp = 0
    cf_tr_hist = []; cf_va_hist = []

    for ep in range(1, CNN_EP + 1):
        convformer.train(); tl = tc = tn = 0
        for X_, y_ in bal_trnl:
            X_, y_ = X_.to(DEVICE), y_.to(DEVICE); cf_opt.zero_grad()
            out = convformer(X_)
            loss = focal_loss(out, y_)
            loss.backward(); nn.utils.clip_grad_norm_(convformer.parameters(), 1.0); cf_opt.step()
            tl += loss.item()*len(y_); tc += (out.argmax(1)==y_).sum().item(); tn += len(y_)
        ta = tc / tn; cf_sch.step()
        convformer.eval()
        with torch.no_grad():
            va = vn = 0
            for X_, y_ in vall:
                X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
                va += (convformer(X_).argmax(1)==y_).sum().item(); vn += len(y_)
        va_acc = va / vn
        cf_tr_hist.append(tl/tn); cf_va_hist.append(va_acc)
        if va_acc > cf_best_va:
            cf_best_va = va_acc; cf_best_w = {k:v.cpu().clone() for k,v in convformer.state_dict().items()}; cf_no_imp = 0
        else:
            cf_no_imp += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:02d}  train {ta*100:.2f}%  val {va_acc*100:.2f}%")
        if cf_no_imp >= PATIENCE:
            print(f"    Early stop at ep {ep}"); break

    convformer.load_state_dict(cf_best_w)
    torch.save(cf_best_w, os.path.join(MODEL_DIR, f"m{mid}_convformer.pt"))
    cf_pred, cf_true, cf_probs = eval_cnn(convformer, tstl)
    cf_acc  = accuracy_score(cf_true, cf_pred)
    cf_f1   = f1_score(cf_true, cf_pred, average="weighted", zero_division=0)
    cf_prec = precision_score(cf_true, cf_pred, average="weighted", zero_division=0)
    cf_rec  = recall_score(cf_true, cf_pred, average="weighted", zero_division=0)
    results["ConvFormer"] = (cf_acc, cf_f1, cf_prec, cf_rec)
    print(f"    Test Acc {cf_acc*100:.2f}%  F1 {cf_f1:.4f}")
    save_confusion_matrix(cf_true, cf_pred, local_names, f"M{mid} ConvFormer",
                          os.path.join(rdir, "convformer_confusion_matrix.png"))
    save_roc(cf_true, cf_probs, local_names, n_cls, f"M{mid} ConvFormer",
             os.path.join(rdir, "convformer_roc.png"))
    save_training_curve(cf_tr_hist, cf_va_hist, f"Mission {mid} ConvFormer",
                        os.path.join(rdir, "convformer_training_curve.png"))
    np.savez_compressed(os.path.join(rdir, "predictions_convformer.npz"),
                        pred=cf_pred, true=cf_true)

    # ---- VAE ----------------------------------------------------------------
    print(f"\n  [VAE] Training {VAE_EP} epochs ...")
    vae = VAE1D(n_feat, WINDOW, LATENT_DIM).to(DEVICE)
    opt_v = optim.AdamW(vae.parameters(), lr=LR, weight_decay=1e-4)
    sch_v = optim.lr_scheduler.CosineAnnealingLR(opt_v, T_max=VAE_EP, eta_min=1e-5)
    vae_losses = []

    for ep in range(1, VAE_EP+1):
        tl = train_vae_epoch(vae, trnl, opt_v); sch_v.step(); vae_losses.append(tl)
        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:02d}/{VAE_EP}  loss {tl:.4f}")

    vae_pt = os.path.join(MODEL_DIR, f"m{mid}_vae.pt")
    torch.save(vae.state_dict(), vae_pt)

    # binary anomaly detection via reconstruction error
    vae.eval()
    with torch.no_grad():
        norm_x = Xtr[ytr == 0]
        if len(norm_x) == 0: norm_x = Xtr[:100]
        n_dl = DataLoader(TelDS(norm_x, np.zeros(len(norm_x), np.int64)), BATCH, shuffle=False, num_workers=0)
        en = []
        for X_, _ in n_dl:
            r, _, _ = vae(X_.to(DEVICE))
            en.extend(F.mse_loss(r, X_.to(DEVICE), reduction="none").mean(dim=(1,2)).cpu().tolist())
        thr = float(np.mean(en) + 2*np.std(en))
        et = []
        for X_, _ in tstl:
            r, _, _ = vae(X_.to(DEVICE))
            et.extend(F.mse_loss(r, X_.to(DEVICE), reduction="none").mean(dim=(1,2)).cpu().tolist())

    et = np.array(et)
    vae_bin_pred = (et > thr).astype(int)
    vae_bin_true = (yte != 0).astype(int)
    vae_acc  = accuracy_score(vae_bin_true, vae_bin_pred)
    vae_f1   = f1_score(vae_bin_true, vae_bin_pred, zero_division=0)
    vae_prec = precision_score(vae_bin_true, vae_bin_pred, zero_division=0)
    vae_rec  = recall_score(vae_bin_true, vae_bin_pred, zero_division=0)
    results["VAE"] = (vae_acc, vae_f1, vae_prec, vae_rec)
    print(f"    Binary Test Acc {vae_acc*100:.2f}%  F1 {vae_f1:.4f}  thr={thr:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(vae_losses, color="#7B1FA2")
    ax.set(title=f"Mission {mid} VAE Training Loss", xlabel="Epoch", ylabel="Loss")
    plt.tight_layout(); plt.savefig(os.path.join(rdir, "vae_loss_curve.png"), dpi=300, bbox_inches="tight"); plt.close()

    # ---- Hybrid -------------------------------------------------------------
    print(f"\n  [Hybrid] Training P1={PHASE1_EP} + P2={PHASE2_EP} epochs ...")
    cnn2 = CNN1D(n_feat, n_cls, DROPOUT).to(DEVICE); cnn2.load_state_dict({k:v.to(DEVICE) for k,v in best_w.items()})
    vae2 = VAE1D(n_feat, WINDOW, LATENT_DIM).to(DEVICE); vae2.load_state_dict(torch.load(vae_pt, map_location=DEVICE))
    hybrid = HybridModel(cnn2, vae2, n_cls).to(DEVICE)

    hist_h = {"tl": [], "va": []}
    hybrid.freeze_backbones()
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, hybrid.parameters()), lr=LR, weight_decay=1e-4)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=PHASE1_EP, eta_min=1e-5)
    best_hva = 0.0; best_hw = None; no_improve_p1 = 0

    for ep in range(1, PHASE1_EP+1):
        tl, ta = train_hybrid_epoch(hybrid, trnl, cw, opt1, phase2=False)
        hybrid.eval()
        with torch.no_grad():
            va = vn = 0
            for X_, y_ in vall:
                X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
                logits, _, _, _ = hybrid(X_)
                va += (logits.argmax(1)==y_).sum().item(); vn += len(y_)
        va_acc = va/vn; sch1.step()
        hist_h["tl"].append(tl); hist_h["va"].append(va_acc)
        if va_acc > best_hva:
            best_hva = va_acc; best_hw = {k:v.cpu().clone() for k,v in hybrid.state_dict().items()}; no_improve_p1 = 0
        else:
            no_improve_p1 += 1
        if ep % 5 == 0 or ep == 1:
            print(f"    P1 Ep {ep:02d}/{PHASE1_EP}  train {ta*100:.2f}%  val {va_acc*100:.2f}%")
        if no_improve_p1 >= PATIENCE:
            print(f"    P1 early stop at ep {ep}")
            break

    hybrid.unfreeze_all()
    opt2 = optim.AdamW(hybrid.parameters(), lr=LR2, weight_decay=1e-4)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EP, eta_min=1e-6)

    no_improve_p2 = 0
    for ep in range(1, PHASE2_EP+1):
        tl, ta = train_hybrid_epoch(hybrid, trnl, cw, opt2, phase2=True)
        hybrid.eval()
        with torch.no_grad():
            va = vn = 0
            for X_, y_ in vall:
                X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
                logits, _, _, _ = hybrid(X_)
                va += (logits.argmax(1)==y_).sum().item(); vn += len(y_)
        va_acc = va/vn; sch2.step()
        hist_h["tl"].append(tl); hist_h["va"].append(va_acc)
        if va_acc > best_hva:
            best_hva = va_acc; best_hw = {k:v.cpu().clone() for k,v in hybrid.state_dict().items()}; no_improve_p2 = 0
        else:
            no_improve_p2 += 1
        if ep % 5 == 0 or ep == 1:
            print(f"    P2 Ep {ep:02d}/{PHASE2_EP}  train {ta*100:.2f}%  val {va_acc*100:.2f}%")
        if no_improve_p2 >= PATIENCE:
            print(f"    P2 early stop at ep {ep}")
            break

    hybrid.load_state_dict(best_hw)
    hyb_pt = os.path.join(MODEL_DIR, f"m{mid}_hybrid.pt")
    torch.save(best_hw, hyb_pt)

    h_pred, h_true, h_probs, h_mus = eval_hybrid(hybrid, tstl)
    hyb_acc  = accuracy_score(h_true, h_pred)
    hyb_f1   = f1_score(h_true, h_pred, average="weighted", zero_division=0)
    hyb_prec = precision_score(h_true, h_pred, average="weighted", zero_division=0)
    hyb_rec  = recall_score(h_true, h_pred, average="weighted", zero_division=0)
    results["Hybrid"] = (hyb_acc, hyb_f1, hyb_prec, hyb_rec)
    print(f"    Test Acc {hyb_acc*100:.2f}%  F1 {hyb_f1:.4f}")

    save_training_curve(hist_h["tl"], hist_h["va"], f"Mission {mid} Hybrid", os.path.join(rdir, "hybrid_training_curve.png"))
    save_confusion_matrix(h_true, h_pred, local_names, f"M{mid} Hybrid", os.path.join(rdir, "hybrid_confusion_matrix.png"))
    save_roc(h_true, h_probs, local_names, n_cls, f"M{mid} Hybrid", os.path.join(rdir, "hybrid_roc.png"))
    save_tsne(h_mus, h_true, local_names, f"Mission {mid}", os.path.join(rdir, "hybrid_tsne.png"))

    # save per-sample predictions for McNemar's statistical tests
    for tag, (pred_arr, true_arr) in [("cnn", (cnn_pred, cnn_true)),
                                       ("bilstm", (lstm_pred, lstm_true)),
                                       ("transformer", (tf_pred, tf_true)),
                                       ("convformer", (cf_pred, cf_true)),
                                       ("hybrid", (h_pred, h_true))]:
        np.savez_compressed(os.path.join(rdir, f"predictions_{tag}.npz"),
                            pred=pred_arr, true=true_arr)

    # ---- per-mission model comparison plot -----------------------------------
    m_labels = ["Accuracy", "Weighted F1", "Precision", "Recall"]
    m_models = list(results.keys())
    m_palette = {"CNN":"#1976D2","BiLSTM":"#7B1FA2","Transformer":"#00838F",
                 "ConvFormer":"#E65100","VAE":"#388E3C","Hybrid":"#AD1457"}
    n_m = len(m_models); w = 0.8 / n_m
    x = np.arange(len(m_labels))
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (mname, vals) in enumerate(results.items()):
        offset = (i - n_m/2 + 0.5) * w
        bars = ax.bar(x + offset, [v*100 for v in vals], w, label=mname,
                      color=m_palette.get(mname, "#607D8B"), alpha=0.85)
        for bar in bars:
            h_ = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h_+0.3, f"{h_:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set(xticks=x, xticklabels=m_labels, ylabel="Score (%)",
           title=f"Mission {mid} — All Models Comparison", ylim=(50, 110))
    ax.axhline(95, color="red", linestyle="--", lw=1.2, alpha=0.7)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(rdir, "model_comparison.png"), dpi=300, bbox_inches="tight"); plt.close()

    # ---- per-mission text report -------------------------------------------
    rpt_lines = []
    rpt_lines.append(f"\n{'='*62}")
    rpt_lines.append(f"  MISSION {mid} - MODEL COMPARISON REPORT")
    rpt_lines.append(f"  Features: {n_feat}   Classes: {uniq_cls}   Test windows: {len(yte):,}")
    rpt_lines.append(f"{'='*62}\n")
    rpt_lines.append(f"  {'Model':<12}  Acc      W-F1     Prec     Rec")
    rpt_lines.append(f"  {'-'*50}")
    for mname, (acc, f1, prec, rec) in results.items():
        rpt_lines.append(f"  {mname:<12}  {acc*100:.2f}%    {f1:.4f}   {prec:.4f}   {rec:.4f}")
    for tag, (tp, tt) in [("CNN", (cnn_pred, cnn_true)), ("BiLSTM", (lstm_pred, lstm_true)),
                           ("Transformer", (tf_pred, tf_true)), ("ConvFormer", (cf_pred, cf_true)),
                           ("Hybrid", (h_pred, h_true))]:
        present = sorted(set(tt) | set(tp))
        rpt_lines.append(f"\n  {tag} Classification Report:\n")
        rpt_lines.append(classification_report(tt, tp, labels=present,
                                               target_names=[local_names[i] for i in present],
                                               digits=4, zero_division=0))
    rpt_text = "\n".join(rpt_lines)
    print(rpt_text)
    with open(os.path.join(rdir, f"m{mid}_report.txt"), "w", encoding="utf-8") as f:
        f.write(rpt_text)

    return results


# ---- cross-mission comparison -----------------------------------------------

def cross_mission_comparison(all_results):
    missions = sorted(all_results.keys())
    models   = ["CNN", "BiLSTM", "Transformer", "ConvFormer", "VAE", "Hybrid"]
    colors   = {"CNN": "#1976D2", "BiLSTM": "#7B1FA2", "Transformer": "#00838F",
                "ConvFormer": "#E65100", "VAE": "#388E3C", "Hybrid": "#AD1457"}
    metrics  = ["Accuracy", "Weighted F1", "Precision", "Recall"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        n_models = len(models); w = 0.75 / n_models
        x = np.arange(len(missions))
        for i, model in enumerate(models):
            vals = []
            for mid in missions:
                r = all_results[mid].get(model, (0,0,0,0))
                vals.append(r[mi] * 100)
            offset = (i - n_models/2 + 0.5) * w
            bars = ax.bar(x + offset, vals, w, label=model, color=colors.get(model, "#607D8B"), alpha=0.85)
            for bar in bars:
                h_ = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, h_+0.4, f"{h_:.1f}", ha="center", va="bottom", fontsize=6.5)
        ax.set(xticks=x, xticklabels=[f"Mission {m}" for m in missions],
               ylabel=f"{metric} (%)", title=f"Cross-Mission {metric}", ylim=(50, 108))
        ax.axhline(95, color="red", linestyle="--", lw=1, alpha=0.7)
        ax.legend(fontsize=8)

    plt.suptitle("Cross-Mission Model Comparison: CNN vs BiLSTM vs Transformer vs Hybrid", fontweight="bold", fontsize=13)
    plt.tight_layout()
    out = os.path.join(REPORT_DIR, "cross_mission_comparison.png")
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"\n  Cross-mission comparison saved -> {out}")


def write_summary(all_results):
    lines = ["\n" + "="*68, "  ALL MISSIONS - MULTI-MODEL PERFORMANCE SUMMARY", "="*68]
    for mid in sorted(all_results.keys()):
        lines.append(f"\n  Mission {mid}:")
        lines.append(f"  {'Model':<12}  Acc      W-F1     Prec     Rec")
        lines.append(f"  {'-'*50}")
        for mname, (acc, f1, prec, rec) in all_results[mid].items():
            lines.append(f"  {mname:<12}  {acc*100:.2f}%    {f1:.4f}   {prec:.4f}   {rec:.4f}")
    txt = "\n".join(lines) + "\n"
    print(txt)
    out = os.path.join(REPORT_DIR, "all_missions_summary.txt")
    with open(out, "w", encoding="utf-8") as f: f.write(txt)
    print(f"  Summary saved -> {out}")


# ---- resume helper ----------------------------------------------------------

def _parse_report_metrics(rpt_txt):
    """Parse acc/f1/prec/rec from a saved m{N}_report.txt file."""
    import re
    results = {}
    try:
        with open(rpt_txt, encoding="utf-8") as f:
            text = f.read()
        # matches lines like:  CNN           99.87%    0.9987   0.9987   0.9987
        for m in re.finditer(r'^\s+(CNN|VAE|Hybrid)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', text, re.MULTILINE):
            model = m.group(1)
            acc   = float(m.group(2)) / 100
            f1    = float(m.group(3))
            prec  = float(m.group(4))
            rec   = float(m.group(5))
            results[model] = (acc, f1, prec, rec)
    except Exception as e:
        print(f"    [WARN] could not parse {rpt_txt}: {e}")
        for model in ("CNN", "VAE", "Hybrid"):
            results[model] = (0, 0, 0, 0)
    return results


# ---- main -------------------------------------------------------------------

def main():
    import random
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    set_style()
    print(f"\nDevice: {DEVICE}")

    # multi-seed training: collect per-seed results then report mean ± std
    seed_all = {}   # seed -> {mid -> {model -> (acc,f1,prec,rec)}}
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        print(f"\n{'#'*68}\n  SEED {seed}\n{'#'*68}")
        seed_results = {}
        for mid in [1, 2, 3]:
            csv_path = os.path.join(DATA_DIR, f"mission{mid}_preprocessed.csv")
            if not os.path.exists(csv_path):
                print(f"  [SKIP] {csv_path} not found"); continue
            seed_results[mid] = run_mission(mid)
        seed_all[seed] = seed_results

    # aggregate mean ± std across seeds
    all_results_mean = {}   # {mid -> {model -> mean_tuple}} for plotting
    missions = sorted({mid for sr in seed_all.values() for mid in sr})
    models_seen = sorted({m for sr in seed_all.values() for res in sr.values() for m in res})

    agg_lines = ["\n" + "="*72,
                 f"  MULTI-SEED RESULTS  (seeds={SEEDS})",
                 "  Mean ± Std over seeds — Weighted metrics", "="*72]
    for mid in missions:
        agg_lines.append(f"\n  Mission {mid}:")
        agg_lines.append(f"  {'Model':<14}  Acc (mean±std)      W-F1 (mean±std)")
        agg_lines.append(f"  {'-'*58}")
        all_results_mean[mid] = {}
        for model in models_seen:
            accs = [seed_all[s][mid][model][0] for s in SEEDS if mid in seed_all[s] and model in seed_all[s][mid]]
            f1s  = [seed_all[s][mid][model][1] for s in SEEDS if mid in seed_all[s] and model in seed_all[s][mid]]
            if not accs: continue
            ma, sa = np.mean(accs)*100, np.std(accs)*100
            mf, sf = np.mean(f1s),  np.std(f1s)
            agg_lines.append(f"  {model:<14}  {ma:.2f} ± {sa:.2f}%          {mf:.4f} ± {sf:.4f}")
            all_results_mean[mid][model] = (np.mean(accs), np.mean(f1s),
                                            np.mean([seed_all[s][mid][model][2] for s in SEEDS if mid in seed_all[s] and model in seed_all[s][mid]]),
                                            np.mean([seed_all[s][mid][model][3] for s in SEEDS if mid in seed_all[s] and model in seed_all[s][mid]]))
    agg_txt = "\n".join(agg_lines) + "\n"
    print(agg_txt)
    agg_out = os.path.join(REPORT_DIR, "multiseed_summary.txt")
    with open(agg_out, "w", encoding="utf-8") as f: f.write(agg_txt)
    print(f"  Multi-seed summary -> {agg_out}")

    if len(all_results_mean) > 1:
        cross_mission_comparison(all_results_mean)
    write_summary(all_results_mean)

    print("\n" + "="*68)
    print("  ALL MISSIONS TRAINING COMPLETE")
    print(f"  Models   -> {MODEL_DIR}")
    print(f"  Reports  -> {REPORT_DIR}")
    print("="*68 + "\n")


if __name__ == "__main__":
    main()

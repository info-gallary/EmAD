"""
Generalized Hybrid CNN-VAE trained on all 3 ESA missions simultaneously.

Uses the zero-padded combined dataset (all_missions_combined.csv).
Evaluates both overall performance and leave-one-mission-out (LOMO) generalization.

Outputs
-------
  models/generalized_hybrid.pt
  reports/generalized/generalized_report.txt
  reports/generalized/generalized_confusion_matrix.png
  reports/generalized/generalized_roc.png
  reports/generalized/generalized_tsne.png
  reports/generalized/generalized_lomo.png   -- leave-one-mission-out bars
  reports/generalized/generalized_training_curve.png
"""

import os
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
COMBINED_CSV = r"d:\UbtVM-Def\Models\data\all_missions_combined.csv"
MODEL_DIR    = r"d:\UbtVM-Def\Models\models"
REPORT_DIR   = r"d:\UbtVM-Def\Models\reports\generalized"
GEN_PT       = r"d:\UbtVM-Def\Models\models\generalized_hybrid.pt"

# ---- hyper-params -----------------------------------------------------------
WINDOW     = 50
STEP       = 2
BATCH      = 512       # larger batch for faster CPU training
PHASE1_EP  = 60
PHASE2_EP  = 80
PATIENCE   = 12
LR1        = 1e-3
LR2        = 5e-5
RUN_LOMO   = True
DROPOUT    = 0.3
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


def make_windows_indexed(X, y, mid_arr, win, step):
    Xw, yw, mw = [], [], []
    for i in range(0, len(X) - win + 1, step):
        seg = y[i:i + win]; v, c = np.unique(seg, return_counts=True)
        Xw.append(X[i:i + win]); yw.append(v[c.argmax()])
        mw.append(mid_arr[i + win // 2])
    return np.array(Xw, np.float32), np.array(yw, np.int64), np.array(mw, np.int64)


class TelDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ---- model definitions (same architecture as train_hybrid.py) ---------------

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
        x = self.cnn.stem(x); x = self.cnn.d1(self.cnn.s1(x))
        x = self.cnn.d2(self.cnn.s2(x)); x = self.cnn.s3(x)
        return self.cnn.pool(x).squeeze(-1)
    def forward(self, x):
        feat = self.cnn_feat(x); recon, mu, lv = self.vae(x)
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


def make_balanced_loader(X, y, batch_size):
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

def train_epoch(model, dl, cw, opt, phase2=False):
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
def eval_model(model, dl):
    model.eval(); ps, ls, prbs, mus = [], [], [], []
    for X, y in dl:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits, _, mu, _ = model(X)
        ps.extend(logits.argmax(1).cpu().tolist())
        ls.extend(y.cpu().tolist())
        prbs.append(torch.softmax(logits, 1).cpu().numpy())
        mus.append(mu.cpu().numpy())
    return np.array(ps), np.array(ls), np.vstack(prbs), np.vstack(mus)


# ---- plots ------------------------------------------------------------------

def save_cm(y_true, y_pred, names, title, out):
    present = sorted(set(y_true) | set(y_pred))
    labs = [names[i] for i in present]
    cm = confusion_matrix(y_true, y_pred, labels=present)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, t in zip(axes, [cm, cmn], ["d", ".2%"], ["Counts", "Row-Norm."]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labs, yticklabels=labs, ax=ax,
                    linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.8})
        ax.set(xlabel="Predicted", ylabel="True", title=f"{title} ({t})")
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle(title, fontweight="bold")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {os.path.basename(out)}")


def save_roc(y_true, probs, names, n_cls, title, out):
    present = sorted(set(y_true))
    yb = label_binarize(y_true, classes=list(range(n_cls)))
    if n_cls == 2:
        yb = np.hstack([1 - yb, yb])
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in present:
        if cls >= yb.shape[1] or yb[:, cls].sum() == 0: continue
        fpr, tpr, _ = roc_curve(yb[:, cls], probs[:, cls])
        ax.plot(fpr, tpr, color=PALETTE[cls % len(PALETTE)], lw=2,
                label=f"{names[cls]}  AUC={auc(fpr, tpr):.3f}")
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title=f"{title} ROC"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {os.path.basename(out)}")


def save_tsne(mus, y_true, names, title, out):
    print("  Running t-SNE ...")
    n = min(len(mus), 5000)
    idx = np.random.choice(len(mus), n, replace=False)
    z2 = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=1000, init="pca").fit_transform(mus[idx])
    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in sorted(set(y_true[idx].tolist())):
        m = y_true[idx] == cls
        ax.scatter(z2[m,0], z2[m,1], c=PALETTE[cls % len(PALETTE)], s=10, alpha=0.6, label=names[cls])
    ax.set(title=f"{title} VAE Latent Space (t-SNE)"); ax.legend(markerscale=2.5)
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {os.path.basename(out)}")


def save_training_curve(tl_hist, va_hist, p1_end, title, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(tl_hist, color="#1565C0"); a1.axvline(p1_end, color="gray", ls=":", lw=1.2, label="Phase 2 start")
    a1.set(title="Loss", xlabel="Epoch"); a1.legend()
    a2.plot([v*100 for v in va_hist], color="#388E3C"); a2.axvline(p1_end, color="gray", ls=":", lw=1.2)
    a2.set(title="Val Accuracy (%)", xlabel="Epoch")
    plt.suptitle(title, fontweight="bold")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {os.path.basename(out)}")


def save_lomo_plot(lomo_results, out):
    missions = sorted(lomo_results.keys())
    accs = [lomo_results[m]["acc"]*100 for m in missions]
    f1s  = [lomo_results[m]["f1"]  for m in missions]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1976D2", "#388E3C", "#E64A19"]
    for ax, vals, metric, ylim in zip(axes,
                                       [accs, f1s],
                                       ["Accuracy (%)", "Weighted F1"],
                                       [(0, 110), (0, 1.1)]):
        bars = ax.bar([f"M{m} (test)" for m in missions], vals, color=colors[:len(missions)], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h*1.01, f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set(ylabel=metric, title=f"LOMO Generalization: {metric}", ylim=ylim)
        if metric == "Accuracy (%)":
            ax.axhline(95, color="red", ls="--", lw=1.2, alpha=0.7, label="95% target")
            ax.legend()
    plt.suptitle("Leave-One-Mission-Out (LOMO) Generalization Test\n(Train on 2 missions, Test on held-out)",
                 fontweight="bold")
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved: {os.path.basename(out)}")


# ---- main -------------------------------------------------------------------

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    set_style()
    print(f"\nDevice: {DEVICE}")

    # -- load combined dataset ------------------------------------------------
    print("Loading combined dataset ...")
    df      = pd.read_csv(COMBINED_CSV, index_col=0)
    fcols   = [c for c in df.columns if c not in ("label","class_name","mission_id")]
    X_full  = df[fcols].values.astype(np.float32)
    y_raw   = df["label"].values.astype(np.int64)
    mid_arr = df["mission_id"].values.astype(np.int64)
    n_feat  = X_full.shape[1]

    # remap labels to contiguous indices
    uniq_cls = sorted(set(y_raw.tolist()))
    remap    = {orig: new for new, orig in enumerate(uniq_cls)}
    y_full   = np.array([remap[v] for v in y_raw], dtype=np.int64)
    n_cls    = len(uniq_cls)
    names    = {new: CLASS_NAMES[orig] for orig, new in remap.items()}

    print(f"  Total samples: {len(X_full):,}  Features: {n_feat}  Classes: {uniq_cls}")
    print(f"  Missions: {dict(zip(*np.unique(mid_arr, return_counts=True)))}")

    # windows with mission tracking
    Xw, yw, mw = make_windows_indexed(X_full, y_full, mid_arr, WINDOW, STEP)
    print(f"  Total windows: {len(Xw):,}")

    # per-class chronological split per mission — guarantees all classes in test
    train_idx, val_idx, test_idx = [], [], []
    for m in sorted(set(mw.tolist())):
        m_idx = np.where(mw == m)[0]
        m_y   = yw[m_idx]
        for cls in np.unique(m_y):
            cls_idx = m_idx[m_y == cls]
            n = len(cls_idx)
            n_te = max(1, int(n * 0.15))
            n_va = max(1, int(n * 0.15))
            test_idx.extend(cls_idx[-n_te:].tolist())
            val_idx.extend(cls_idx[-(n_te + n_va):-n_te].tolist())
            train_idx.extend(cls_idx[:-(n_te + n_va)].tolist())
    train_idx = sorted(train_idx); val_idx = sorted(val_idx); test_idx = sorted(test_idx)
    Xtr, ytr, mtr = Xw[train_idx], yw[train_idx], mw[train_idx]
    Xva, yva      = Xw[val_idx],   yw[val_idx]
    Xte, yte, mte = Xw[test_idx],  yw[test_idx], mw[test_idx]
    print(f"  Train:{len(Xtr):,}  Val:{len(Xva):,}  Test:{len(Xte):,}")
    print(f"  Test class dist: { {c: int((yte==c).sum()) for c in np.unique(yte)} }")

    cw   = class_weights(ytr, n_cls, DEVICE)
    trnl = DataLoader(TelDS(Xtr, ytr), BATCH, shuffle=True, num_workers=0)
    vall = DataLoader(TelDS(Xva, yva), BATCH, shuffle=False, num_workers=0)
    tstl = DataLoader(TelDS(Xte, yte), BATCH, shuffle=False, num_workers=0)

    # -- build model ----------------------------------------------------------
    cnn    = CNN1D(n_feat, n_cls, DROPOUT).to(DEVICE)
    vae    = VAE1D(n_feat, WINDOW, LATENT_DIM).to(DEVICE)
    hybrid = HybridModel(cnn, vae, n_cls).to(DEVICE)
    total_p = sum(p.numel() for p in hybrid.parameters())
    print(f"\nGeneralized model parameters: {total_p:,}")

    tl_hist = []; va_hist = []; p1_end = PHASE1_EP

    if os.path.exists(GEN_PT):
        print(f"\n  [RESUME] Loading saved model from {GEN_PT}")
        hybrid.load_state_dict(torch.load(GEN_PT, map_location=DEVICE))
    else:
        best_va = 0.0; best_w = None

        # -- Phase 1: meta-learner warmup -----------------------------------------
        print(f"\n{'='*60}\n  Phase 1: Meta-learner warmup (max {PHASE1_EP} epochs, patience={PATIENCE})\n{'='*60}")
        hybrid.freeze_backbones()
        opt1 = optim.AdamW(filter(lambda p: p.requires_grad, hybrid.parameters()), lr=LR1, weight_decay=1e-4)
        sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=PHASE1_EP, eta_min=1e-5)
        no_improve_p1 = 0

        for ep in range(1, PHASE1_EP+1):
            tl, ta = train_epoch(hybrid, trnl, cw, opt1, phase2=False)
            hybrid.eval()
            with torch.no_grad():
                va = vn = 0
                for X_, y_ in vall:
                    X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
                    logits, _, _, _ = hybrid(X_)
                    va += (logits.argmax(1)==y_).sum().item(); vn += len(y_)
            va_acc = va/vn; sch1.step()
            tl_hist.append(tl); va_hist.append(va_acc)
            if va_acc > best_va:
                best_va = va_acc; best_w = {k:v.cpu().clone() for k,v in hybrid.state_dict().items()}; no_improve_p1 = 0
            else:
                no_improve_p1 += 1
            if ep % 5 == 0 or ep == 1:
                print(f"  Ep {ep:02d}/{PHASE1_EP}  train {ta*100:.2f}%  val {va_acc*100:.2f}%")
            if no_improve_p1 >= PATIENCE:
                print(f"  P1 early stop at ep {ep}"); break

        # -- Phase 2: joint fine-tune ---------------------------------------------
        print(f"\n{'='*60}\n  Phase 2: Joint fine-tune (max {PHASE2_EP} epochs, patience={PATIENCE})\n{'='*60}")
        hybrid.unfreeze_all()
        opt2 = optim.AdamW(hybrid.parameters(), lr=LR2, weight_decay=1e-4)
        sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EP, eta_min=1e-6)
        no_improve_p2 = 0

        for ep in range(1, PHASE2_EP+1):
            tl, ta = train_epoch(hybrid, trnl, cw, opt2, phase2=True)
            hybrid.eval()
            with torch.no_grad():
                va = vn = 0
                for X_, y_ in vall:
                    X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
                    logits, _, _, _ = hybrid(X_)
                    va += (logits.argmax(1)==y_).sum().item(); vn += len(y_)
            va_acc = va/vn; sch2.step()
            tl_hist.append(tl); va_hist.append(va_acc)
            if va_acc > best_va:
                best_va = va_acc; best_w = {k:v.cpu().clone() for k,v in hybrid.state_dict().items()}; no_improve_p2 = 0
            else:
                no_improve_p2 += 1
            if ep % 5 == 0 or ep == 1:
                print(f"  Ep {ep:02d}/{PHASE2_EP}  train {ta*100:.2f}%  val {va_acc*100:.2f}%  lr={sch2.get_last_lr()[0]:.1e}")
            if no_improve_p2 >= PATIENCE:
                print(f"  P2 early stop at ep {ep}"); break

        hybrid.load_state_dict(best_w)
        torch.save(best_w, GEN_PT)
        print(f"\n  Best val acc: {best_va*100:.2f}%  -> {GEN_PT}")

    # -- evaluation on full test set ------------------------------------------
    pred, true, probs, mus = eval_model(hybrid, tstl)
    acc  = accuracy_score(true, pred)
    f1w  = f1_score(true, pred, average="weighted", zero_division=0)
    f1m  = f1_score(true, pred, average="macro",    zero_division=0)
    prec = precision_score(true, pred, average="weighted", zero_division=0)
    rec  = recall_score(true, pred,    average="weighted", zero_division=0)
    print(f"\n  Test Acc {acc*100:.2f}%  W-F1 {f1w:.4f}  Macro-F1 {f1m:.4f}")

    # -- plots ----------------------------------------------------------------
    print(f"\n{'='*60}\n  Generating plots\n{'='*60}")
    save_training_curve(tl_hist, va_hist, p1_end, "Generalized Hybrid CNN-VAE", os.path.join(REPORT_DIR, "generalized_training_curve.png"))
    save_cm(true, pred, names, "Generalized Model", os.path.join(REPORT_DIR, "generalized_confusion_matrix.png"))
    save_roc(true, probs, names, n_cls, "Generalized Model", os.path.join(REPORT_DIR, "generalized_roc.png"))
    save_tsne(mus, true, names, "Generalized Model", os.path.join(REPORT_DIR, "generalized_tsne.png"))

    # -- per-mission test breakdown -------------------------------------------
    print("\n  Per-mission performance on test set:")
    per_mission = {}
    for m in sorted(set(mte.tolist())):
        mask = mte == m
        if mask.sum() == 0: continue
        # test windows with mission mask
        m_ds = TelDS(Xte[mask], yte[mask])
        m_dl = DataLoader(m_ds, BATCH, shuffle=False, num_workers=0)
        mp, mt, mpr, _ = eval_model(hybrid, m_dl)
        m_acc = accuracy_score(mt, mp)
        m_f1  = f1_score(mt, mp, average="weighted", zero_division=0)
        per_mission[m] = {"acc": m_acc, "f1": m_f1}
        print(f"    Mission {m}: Acc {m_acc*100:.2f}%  F1 {m_f1:.4f}  (n_windows={mask.sum()})")

    # -- Leave-One-Mission-Out (LOMO) evaluation -------------------------------
    lomo_results = {}
    if RUN_LOMO:
        print(f"\n{'='*60}\n  Leave-One-Mission-Out (LOMO) Generalization Test\n{'='*60}")
        all_missions = sorted(set(mw.tolist()))
        for held_out in all_missions:
            print(f"\n  Held-out mission: {held_out}")
            mask_train = mw != held_out
            mask_test  = mw == held_out

            X_ltr = Xw[mask_train]; y_ltr = yw[mask_train]
            X_lte = Xw[mask_test];  y_lte = yw[mask_test]

            if len(X_ltr) < 100 or len(X_lte) < 10:
                print(f"    [SKIP] too few samples")
                continue

            n_l = len(X_ltr)
            Xl2, yl2 = X_ltr[:int(0.85 * n_l)], y_ltr[:int(0.85 * n_l)]
            Xv2, yv2 = X_ltr[int(0.85 * n_l):], y_ltr[int(0.85 * n_l):]

            cw_l  = class_weights(yl2, n_cls, DEVICE)
            lt_dl = DataLoader(TelDS(Xl2, yl2), BATCH, shuffle=True, num_workers=0)
            lv_dl = DataLoader(TelDS(Xv2, yv2), BATCH, shuffle=False, num_workers=0)
            le_dl = DataLoader(TelDS(X_lte, y_lte), BATCH, shuffle=False, num_workers=0)

            l_cnn = CNN1D(n_feat, n_cls, DROPOUT).to(DEVICE)
            l_vae = VAE1D(n_feat, WINDOW, LATENT_DIM).to(DEVICE)
            l_hyb = HybridModel(l_cnn, l_vae, n_cls).to(DEVICE)
            l_hyb.freeze_backbones()
            l_opt = optim.AdamW(filter(lambda p: p.requires_grad, l_hyb.parameters()), lr=LR1, weight_decay=1e-4)

            best_lva = 0.0; best_lw = None
            for ep in range(1, 8):
                train_epoch(l_hyb, lt_dl, cw_l, l_opt, phase2=False)
                l_hyb.eval()
                with torch.no_grad():
                    va = vn = 0
                    for X_, y_ in lv_dl:
                        X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
                        logits, _, _, _ = l_hyb(X_)
                        va += (logits.argmax(1)==y_).sum().item(); vn += len(y_)
                va_acc = va/vn
                if va_acc > best_lva: best_lva = va_acc; best_lw = {k:v.cpu().clone() for k,v in l_hyb.state_dict().items()}

            l_hyb.unfreeze_all()
            l_opt2 = optim.AdamW(l_hyb.parameters(), lr=LR2, weight_decay=1e-4)
            for ep in range(1, 12):
                train_epoch(l_hyb, lt_dl, cw_l, l_opt2, phase2=True)
                l_hyb.eval()
                with torch.no_grad():
                    va = vn = 0
                    for X_, y_ in lv_dl:
                        X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
                        logits, _, _, _ = l_hyb(X_)
                        va += (logits.argmax(1)==y_).sum().item(); vn += len(y_)
                va_acc = va/vn
                if va_acc > best_lva: best_lva = va_acc; best_lw = {k:v.cpu().clone() for k,v in l_hyb.state_dict().items()}

            if best_lw: l_hyb.load_state_dict(best_lw)
            lp, lt, *_ = eval_model(l_hyb, le_dl)
            l_acc = accuracy_score(lt, lp)
            l_f1  = f1_score(lt, lp, average="weighted", zero_division=0)
            lomo_results[held_out] = {"acc": l_acc, "f1": l_f1}
            print(f"    Mission {held_out} (test-only): Acc {l_acc*100:.2f}%  F1 {l_f1:.4f}")

        if lomo_results:
            save_lomo_plot(lomo_results, os.path.join(REPORT_DIR, "generalized_lomo.png"))
    else:
        print("\n[LOMO skipped — RUN_LOMO=False]")

    # -- text report ----------------------------------------------------------
    present = sorted(set(true) | set(pred))
    clf_rpt = classification_report(true, pred,
                                     labels=present,
                                     target_names=[names[i] for i in present],
                                     digits=4, zero_division=0)

    lomo_lines = "\n".join(
        f"  Mission {m}: Acc {v['acc']*100:.2f}%  F1 {v['f1']:.4f}"
        for m, v in sorted(lomo_results.items()))

    per_m_lines = "\n".join(
        f"  Mission {m}: Acc {v['acc']*100:.2f}%  F1 {v['f1']:.4f}"
        for m, v in sorted(per_mission.items()))

    report = (
        "\n" + "="*68 + "\n"
        "  GENERALIZED HYBRID CNN-VAE - CROSS-MISSION REPORT\n"
        "  Trained on all 3 ESA missions (zero-padded to 275 features)\n"
        "  Architecture: CNN Residual + VAE latent -> MLP meta-learner\n"
        + "="*68 + "\n\n"
        f"  Overall Test Accuracy     : {acc*100:.2f}%\n"
        f"  Weighted F1-Score         : {f1w:.4f}\n"
        f"  Macro F1-Score            : {f1m:.4f}\n"
        f"  Weighted Precision        : {prec:.4f}\n"
        f"  Weighted Recall           : {rec:.4f}\n\n"
        + "-"*68 + "\n"
        "  Per-Mission Breakdown (test windows):\n\n"
        + per_m_lines + "\n\n"
        + "-"*68 + "\n"
        "  Leave-One-Mission-Out Generalization:\n\n"
        + lomo_lines + "\n\n"
        + "-"*68 + "\n"
        "  Classification Report:\n\n"
        + clf_rpt + "\n"
    )
    print(report)
    with open(os.path.join(REPORT_DIR, "generalized_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved -> {REPORT_DIR}/generalized_report.txt")

    print("\n" + "="*68)
    print(f"  GENERALIZED MODEL COMPLETE")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Model    -> {GEN_PT}")
    print(f"  Reports  -> {REPORT_DIR}")
    print("="*68 + "\n")


if __name__ == "__main__":
    main()

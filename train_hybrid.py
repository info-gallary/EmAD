"""
Hybrid CNN-VAE Anomaly Detector with Meta-Learner
==================================================

Architecture
------------
  Input (275 x 50)
      |                              |
  CNN Backbone                  VAE Encoder
  stem->s1->d1->s2->d2->s3      enc->AdaptPool->mu_l
      |                              |
  Global AvgPool (256-d)        Latent mu (64-d) + lv (64-d)
      |                              |     |
      +------------- concat ---------+     +-> Decoder -> recon
                         |
                    Meta-Learner MLP
                    320 -> 256 -> 128 -> n_cls

Training strategy
-----------------
  Phase 1 (warmup, 20 epochs, lr=1e-3):
      Freeze CNN backbone + VAE; train meta-learner only.
  Phase 2 (joint fine-tune, 30 epochs, lr=5e-5):
      Unfreeze all; joint loss = w_cls*CE + w_rec*MSE + w_kl*KL

Publication plots (saved to reports/hybrid/)
--------------------------------------------
  1. hybrid_loss_curves.png       -- total / cls / recon / KL per epoch
  2. hybrid_confusion_matrix.png  -- normalised, %-annotated
  3. hybrid_tsne.png              -- t-SNE of VAE latent space by class
  4. hybrid_roc.png               -- per-class ROC with AUC (one-vs-rest)
  5. hybrid_pr.png                -- per-class Precision-Recall with AP
  6. model_comparison.png         -- CNN vs VAE vs Hybrid grouped bars
  7. hybrid_recon_dist.png        -- reconstruction MSE distribution by class
  8. hybrid_calibration.png       -- reliability diagram (confidence vs accuracy)
"""

import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score,
    classification_report, confusion_matrix,
    f1_score, precision_recall_curve,
    precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ---- paths ------------------------------------------------------------------
CSV_PATH    = r"d:\UbtVM-Def\Models\preprocessed_dataset.csv"
CNN_PT      = r"d:\UbtVM-Def\Models\cnn1d_anomaly.pt"
VAE_PT      = r"d:\UbtVM-Def\Models\vae_anomaly.pt"
HYBRID_PT   = r"d:\UbtVM-Def\Models\hybrid_anomaly.pt"
REPORT_DIR  = r"d:\UbtVM-Def\Models\reports\hybrid"

# ---- hyper-params -----------------------------------------------------------
WINDOW     = 50
STEP       = 2
BATCH      = 256
PHASE1_EP  = 20      # meta-learner warmup (backbones frozen)
PHASE2_EP  = 30      # joint fine-tune
LR1        = 1e-3    # phase-1 lr
LR2        = 5e-5    # phase-2 lr
DROPOUT    = 0.3
LATENT_DIM = 64
W_CLS      = 1.0     # classification loss weight
W_REC      = 0.3     # reconstruction loss weight
W_KL       = 0.05    # KL divergence weight
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Normal", "Comm.", "Power", "Thermal",
               "Software", "Rare-Event", "Comm-Gap", "Unknown"]
PALETTE     = ["#2196F3", "#FF5722", "#9C27B0", "#F44336",
               "#FF9800", "#4CAF50", "#00BCD4", "#607D8B"]


# ---- publication style ------------------------------------------------------

def set_style():
    plt.rcParams.update({
        "figure.dpi"        : 150,
        "savefig.dpi"       : 300,
        "font.family"       : "DejaVu Sans",
        "font.size"         : 11,
        "axes.labelsize"    : 12,
        "axes.titlesize"    : 13,
        "axes.titleweight"  : "bold",
        "xtick.labelsize"   : 10,
        "ytick.labelsize"   : 10,
        "legend.fontsize"   : 9,
        "legend.framealpha" : 0.9,
        "figure.facecolor"  : "white",
        "axes.facecolor"    : "#f9f9f9",
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "axes.grid"         : True,
        "grid.alpha"        : 0.35,
        "grid.linestyle"    : "--",
        "lines.linewidth"   : 2.0,
    })


# ---- dataset ----------------------------------------------------------------

def make_windows(X, y, win, step):
    Xw, yw = [], []
    for i in range(0, len(X) - win + 1, step):
        seg = y[i:i + win]; v, c = np.unique(seg, return_counts=True)
        Xw.append(X[i:i + win]); yw.append(v[c.argmax()])
    return np.array(Xw, np.float32), np.array(yw, np.int64)


class TelDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ---- backbone definitions (must match train_cnn1d.py exactly) ---------------

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


# ---- hybrid model -----------------------------------------------------------

class HybridModel(nn.Module):
    """
    Meta-learner that fuses CNN backbone features with VAE latent space.
    CNN and VAE are sub-modules; only meta-learner is new.
    """
    def __init__(self, cnn: CNN1D, vae: VAE1D, n_cls: int):
        super().__init__()
        self.cnn = cnn
        self.vae = vae
        # fusion dimension: CNN pool (256) + VAE mu (64) = 320
        self.meta = nn.Sequential(
            nn.Linear(320, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, n_cls),
        )

    def cnn_feat(self, x):
        x = self.cnn.stem(x)
        x = self.cnn.d1(self.cnn.s1(x))
        x = self.cnn.d2(self.cnn.s2(x))
        x = self.cnn.s3(x)
        return self.cnn.pool(x).squeeze(-1)          # (B, 256)

    def forward(self, x):
        feat          = self.cnn_feat(x)              # (B, 256)
        recon, mu, lv = self.vae(x)                   # (B,nf,w), (B,64), (B,64)
        logits        = self.meta(torch.cat([feat, mu], dim=1))
        return logits, recon, mu, lv

    def freeze_backbones(self):
        for p in self.cnn.parameters(): p.requires_grad_(False)
        for p in self.vae.parameters(): p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad_(True)


# ---- losses -----------------------------------------------------------------

def joint_loss(logits, y, recon, x, mu, lv, cw):
    cls_l = F.cross_entropy(logits, y, weight=cw, label_smoothing=0.05)
    rec_l = F.mse_loss(recon, x)
    kl_l  = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    total = W_CLS * cls_l + W_REC * rec_l + W_KL * kl_l
    return total, cls_l.item(), rec_l.item(), kl_l.item()


# ---- train / eval loops -----------------------------------------------------

def train_epoch(model, dl, cw, opt, dev, phase2=False):
    model.train()
    tl = tc = tn = 0
    tc_l = tr_l = tk_l = 0
    for X, y in dl:
        X, y = X.to(dev), y.to(dev)
        opt.zero_grad()
        logits, recon, mu, lv = model(X)
        if phase2:
            loss, cl, rl, kl = joint_loss(logits, y, recon, X, mu, lv, cw)
            tc_l += cl * len(y); tr_l += rl * len(y); tk_l += kl * len(y)
        else:
            loss = F.cross_entropy(logits, y, weight=cw, label_smoothing=0.05)
            tc_l += loss.item() * len(y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item() * len(y)
        tc += (logits.argmax(1) == y).sum().item()
        tn += len(y)
    return tl/tn, tc/tn, tc_l/tn, tr_l/tn, tk_l/tn


@torch.no_grad()
def eval_epoch(model, dl, cw, dev):
    model.eval()
    tl = tc = tn = 0
    ps, ls, probs = [], [], []
    mus = []
    for X, y in dl:
        X, y = X.to(dev), y.to(dev)
        logits, recon, mu, lv = model(X)
        loss, *_ = joint_loss(logits, y, recon, X, mu, lv, cw)
        p = logits.argmax(1)
        tc += (p == y).sum().item(); tn += len(y)
        tl += loss.item() * len(y)
        ps.extend(p.cpu().tolist()); ls.extend(y.cpu().tolist())
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        mus.append(mu.cpu().numpy())
    return tl/tn, tc/tn, np.array(ps), np.array(ls), np.vstack(probs), np.vstack(mus)


@torch.no_grad()
def get_recon_errors(model, dl, dev):
    model.eval()
    errs = []
    for X, _ in dl:
        X = X.to(dev)
        _, recon, _, _ = model(X)
        mse = F.mse_loss(recon, X, reduction="none").mean(dim=(1, 2))
        errs.extend(mse.cpu().tolist())
    return np.array(errs)


# ---- evaluation of standalone CNN and VAE (same test split) -----------------

@torch.no_grad()
def eval_standalone_cnn(cnn, dl, cw, dev):
    cnn.eval()
    ps, ls, probs = [], [], []
    for X, y in dl:
        X, y = X.to(dev), y.to(dev)
        out = cnn(X); p = out.argmax(1)
        ps.extend(p.cpu().tolist()); ls.extend(y.cpu().tolist())
        probs.append(torch.softmax(out, 1).cpu().numpy())
    return np.array(ps), np.array(ls), np.vstack(probs)


@torch.no_grad()
def eval_standalone_vae(vae, normal_train, test_dl, dev):
    vae.eval()
    # threshold from normal training windows
    n_dl = DataLoader(TelDS(normal_train, np.zeros(len(normal_train), np.int64)),
                      256, shuffle=False, num_workers=0)
    en = []
    for X, _ in n_dl:
        X = X.to(dev); r, _, _ = vae(X)
        en.extend(F.mse_loss(r, X, reduction="none").mean(dim=(1,2)).cpu().tolist())
    thr = float(np.mean(en) + 2 * np.std(en))
    et = []
    yt = []
    for X, y in test_dl:
        X = X.to(dev); r, _, _ = vae(X)
        et.extend(F.mse_loss(r, X, reduction="none").mean(dim=(1,2)).cpu().tolist())
        yt.extend(y.tolist())
    et = np.array(et); yt = np.array(yt)
    return et, yt, thr


# ---- plots ------------------------------------------------------------------

def plot_loss_curves(h, out):
    fig = plt.figure(figsize=(16, 4))
    gs  = gridspec.GridSpec(1, 4, figure=fig)
    titles = ["Total Loss", "Classification Loss", "Reconstruction MSE", "KL Divergence"]
    keys   = [("tl","vl"), ("cl",""), ("rl",""), ("kl","")]
    colors = [("#1565C0","#EF5350"), ("#1565C0",""), ("#388E3C",""), ("#7B1FA2","")]
    ep = range(1, len(h["tl"]) + 1)
    for i, (title, (tk, vk), (tc, vc)) in enumerate(zip(titles, keys, colors)):
        ax = fig.add_subplot(gs[i])
        ax.plot(ep, h[tk], color=tc, label="Train")
        if vk and vk in h: ax.plot(ep, h[vk], color=vc, linestyle="--", label="Val")
        p1 = len(h.get("phase1_end", []))
        if p1:
            ax.axvline(p1, color="gray", linestyle=":", linewidth=1.2, label="Phase2 start")
        ax.set(title=title, xlabel="Epoch"); ax.legend(fontsize=8)
    plt.suptitle("Hybrid CNN-VAE: Training Loss Breakdown", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved {os.path.basename(out)}")


def plot_confusion_matrix(y_true, y_pred, names, out):
    present = sorted(set(y_true) | set(y_pred))
    labs    = [names[i] for i in present]
    cm_raw  = confusion_matrix(y_true, y_pred, labels=present)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
            axes,
            [cm_raw, cm_norm],
            ["d",    ".2%"],
            ["Counts", "Row-Normalised"]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labs, yticklabels=labs, ax=ax,
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.8})
        ax.set(xlabel="Predicted", ylabel="True",
               title=f"Confusion Matrix ({title})")
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle("Hybrid CNN-VAE Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved {os.path.basename(out)}")


def plot_tsne(mus, y_true, names, palette, out):
    print("  Running t-SNE on latent vectors ...")
    z2 = TSNE(n_components=2, perplexity=40, random_state=42,
              n_iter=1000, init="pca").fit_transform(mus)

    present = sorted(set(y_true))
    fig, ax  = plt.subplots(figsize=(9, 7))
    for cls in present:
        idx = y_true == cls
        ax.scatter(z2[idx, 0], z2[idx, 1],
                   c=palette[cls], s=12, alpha=0.65, edgecolors="none",
                   label=names[cls])
    ax.set(title="t-SNE of VAE Latent Space (Hybrid, Test Set)",
           xlabel="t-SNE dim 1", ylabel="t-SNE dim 2")
    ax.legend(markerscale=2.5, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved {os.path.basename(out)}")


def plot_roc(y_true, probs, names, palette, n_cls, out):
    present = sorted(set(y_true))
    yb = label_binarize(y_true, classes=list(range(n_cls)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in present:
        if yb[:, cls].sum() == 0: continue
        fpr, tpr, _ = roc_curve(yb[:, cls], probs[:, cls])
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=palette[cls], lw=2,
                label=f"{names[cls]}  (AUC={a:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Per-Class ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved {os.path.basename(out)}")


def plot_pr(y_true, probs, names, palette, n_cls, out):
    present = sorted(set(y_true))
    yb = label_binarize(y_true, classes=list(range(n_cls)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in present:
        if yb[:, cls].sum() == 0: continue
        prec, rec, _ = precision_recall_curve(yb[:, cls], probs[:, cls])
        ap = average_precision_score(yb[:, cls], probs[:, cls])
        ax.plot(rec, prec, color=palette[cls], lw=2,
                label=f"{names[cls]}  (AP={ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Per-Class Precision-Recall Curves")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved {os.path.basename(out)}")


def plot_model_comparison(metrics_dict, out):
    """
    metrics_dict: {'CNN': (acc,f1,prec,rec), 'VAE': (...), 'Hybrid': (...)}
    """
    labels  = ["Accuracy", "Weighted F1", "Precision", "Recall"]
    models  = list(metrics_dict.keys())
    colors  = ["#1976D2", "#388E3C", "#E64A19"]
    x       = np.arange(len(labels))
    w       = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (model, vals) in enumerate(metrics_dict.items()):
        bars = ax.bar(x + (i - 1) * w, [v * 100 for v in vals],
                      w, label=model, color=colors[i], alpha=0.85,
                      edgecolor="white", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set(xticks=x, xticklabels=labels, ylabel="Score (%)",
           title="Model Comparison: CNN vs VAE vs Hybrid Meta-Learner",
           ylim=(80, 105))
    ax.legend(loc="lower right")
    ax.axhline(95, color="red", linestyle="--", lw=1.2, alpha=0.7, label="95% target")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved {os.path.basename(out)}")


def plot_recon_distribution(errors, y_true, names, palette, out):
    present = sorted(set(y_true))
    fig, ax = plt.subplots(figsize=(10, 5))
    for cls in present:
        e = errors[y_true == cls]
        ax.hist(e, bins=50, alpha=0.55, label=f"{names[cls]} (n={len(e)})",
                color=palette[cls], density=True, edgecolor="none")
    ax.set(xlabel="Reconstruction MSE", ylabel="Density",
           title="VAE Reconstruction Error Distribution by Class (Hybrid, Test Set)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved {os.path.basename(out)}")


def plot_calibration(y_true, probs, names, n_cls, out):
    present = sorted(set(y_true))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(present)))
    yb = label_binarize(y_true, classes=list(range(n_cls)))
    for cls, col in zip(present, colors):
        if yb[:, cls].sum() < 5: continue
        frac_pos, mean_pred = calibration_curve(yb[:, cls], probs[:, cls], n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", lw=1.8, color=col, label=names[cls])
    ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
           title="Calibration Plot (Reliability Diagram)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved {os.path.basename(out)}")


# ---- metrics report ---------------------------------------------------------

def write_report(y_true, y_pred, probs, names, n_cls, metrics_dict, out):
    present = sorted(set(y_true))
    labs    = [names[i] for i in present]
    acc  = accuracy_score(y_true, y_pred)
    f1w  = f1_score(y_true, y_pred, average="weighted", labels=present, zero_division=0)
    f1m  = f1_score(y_true, y_pred, average="macro",    labels=present, zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", labels=present, zero_division=0)
    rec  = recall_score(y_true, y_pred,    average="weighted", labels=present, zero_division=0)
    rpt  = classification_report(y_true, y_pred, labels=present,
                                  target_names=labs, digits=4, zero_division=0)
    yb   = label_binarize(y_true, classes=list(range(n_cls)))
    aucs = {}
    for cls in present:
        if yb[:, cls].sum() > 0:
            aucs[names[cls]] = roc_auc_score(yb[:, cls], probs[:, cls])

    cmp_lines = ""
    for model, (ma, mf, mp, mr) in metrics_dict.items():
        cmp_lines += f"  {model:<10s}  Acc {ma*100:6.2f}%  F1 {mf:.4f}  Prec {mp:.4f}  Rec {mr:.4f}\n"

    txt = (
        "\n" + "=" * 68 + "\n"
        "  ESA ANOMALY DETECTION - HYBRID CNN-VAE META-LEARNER REPORT\n"
        "  Architecture: CNN Backbone (256-d) + VAE Latent (64-d) -> MLP\n"
        "  Training: Phase-1 warmup (20ep) + Phase-2 joint (30ep)\n"
        + "=" * 68 + "\n\n"
        f"  Accuracy (overall)    : {acc * 100:.2f}%\n"
        f"  Weighted F1-Score     : {f1w:.4f}\n"
        f"  Macro F1-Score        : {f1m:.4f}\n"
        f"  Weighted Precision    : {prec:.4f}\n"
        f"  Weighted Recall       : {rec:.4f}\n\n"
        + "-" * 68 + "\n"
        "  Per-Class AUC (one-vs-rest):\n"
        + "".join(f"    {k:<20s} : {v:.4f}\n" for k, v in aucs.items()) + "\n"
        + "-" * 68 + "\n"
        "  Per-Class Classification Report:\n\n"
        + rpt + "\n"
        + "-" * 68 + "\n"
        "  Cross-Model Comparison (same stratified test split):\n\n"
        + cmp_lines + "\n"
    )
    print(txt)
    with open(out, "w", encoding="utf-8") as f: f.write(txt)
    print(f"  Saved {os.path.basename(out)}")
    return acc, f1w


# ---- main -------------------------------------------------------------------

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    set_style()
    print(f"\nDevice: {DEVICE}")

    # -- load data ------------------------------------------------------------
    print("Loading CSV ...")
    df    = pd.read_csv(CSV_PATH, index_col="timestamp")
    fc    = [c for c in df.columns if c not in ("label", "class_name")]
    X     = df[fc].values.astype(np.float32)
    y     = df["label"].values.astype(np.int64)
    n_cls = int(y.max()) + 1
    n_feat = X.shape[1]
    print(f"  Samples:{len(X):,}  Features:{n_feat}  Classes:{np.unique(y).tolist()}")

    Xw, yw = make_windows(X, y, WINDOW, STEP)
    n=len(Xw); n_tr,n_va=int(0.70*n),int(0.85*n)
    Xtr,ytr=Xw[:n_tr],yw[:n_tr]
    Xva,yva=Xw[n_tr:n_va],yw[n_tr:n_va]
    Xte,yte=Xw[n_va:],yw[n_va:]
    print(f"  Windows  Train:{len(Xtr):,}  Val:{len(Xva):,}  Test:{len(Xte):,}")

    ci, cn = np.unique(ytr, return_counts=True)
    wts    = np.ones(n_cls, np.float32)
    for c, n in zip(ci, cn): wts[c] = len(ytr) / (len(ci) * n)
    cw = torch.tensor(wts, dtype=torch.float32).to(DEVICE)

    trnl = DataLoader(TelDS(Xtr, ytr), BATCH, shuffle=True,  num_workers=0)
    vall = DataLoader(TelDS(Xva, yva), BATCH, shuffle=False, num_workers=0)
    tstl = DataLoader(TelDS(Xte, yte), BATCH, shuffle=False, num_workers=0)

    # -- load pre-trained backbones ------------------------------------------
    print("\nLoading pre-trained CNN and VAE ...")
    cnn = CNN1D(n_feat, n_cls, DROPOUT).to(DEVICE)
    vae = VAE1D(n_feat, WINDOW, LATENT_DIM).to(DEVICE)
    if os.path.exists(CNN_PT):
        cnn.load_state_dict(torch.load(CNN_PT, map_location=DEVICE))
        print("  Loaded cnn1d_anomaly.pt")
    else:
        print("  [WARN] CNN weights not found; initialising randomly.")
    if os.path.exists(VAE_PT):
        vae.load_state_dict(torch.load(VAE_PT, map_location=DEVICE))
        print("  Loaded vae_anomaly.pt")
    else:
        print("  [WARN] VAE weights not found; initialising randomly.")

    # standalone evaluations for comparison
    cnn_pred, cnn_true, cnn_probs = eval_standalone_cnn(cnn, tstl, cw, DEVICE)
    Xnorm = Xtr[ytr == 0]
    vae_errors, vae_true, vae_thr = eval_standalone_vae(vae, Xnorm, tstl, DEVICE)
    vae_bin_pred = (vae_errors > vae_thr).astype(int)
    vae_bin_true = (vae_true != 0).astype(int)

    metrics_cnn = (accuracy_score(cnn_true, cnn_pred),
                   f1_score(cnn_true, cnn_pred, average="weighted", zero_division=0),
                   precision_score(cnn_true, cnn_pred, average="weighted", zero_division=0),
                   recall_score(cnn_true, cnn_pred, average="weighted", zero_division=0))
    metrics_vae = (accuracy_score(vae_bin_true, vae_bin_pred),
                   f1_score(vae_bin_true, vae_bin_pred, zero_division=0),
                   precision_score(vae_bin_true, vae_bin_pred, zero_division=0),
                   recall_score(vae_bin_true, vae_bin_pred, zero_division=0))

    # -- build hybrid ---------------------------------------------------------
    hybrid = HybridModel(cnn, vae, n_cls).to(DEVICE)
    total_p = sum(p.numel() for p in hybrid.parameters() if p.requires_grad)
    meta_p  = sum(p.numel() for p in hybrid.meta.parameters())
    print(f"\nHybrid parameters : {total_p:,}  (meta-learner: {meta_p:,})")

    hist = {"tl": [], "vl": [], "va": [], "cl": [], "rl": [], "kl": []}

    # ---- Phase 1: freeze backbones, train meta-learner ----------------------
    print(f"\n{'='*60}\n  Phase 1: Meta-learner warmup ({PHASE1_EP} epochs, backbones frozen)\n{'='*60}")
    hybrid.freeze_backbones()
    opt1  = optim.AdamW(filter(lambda p: p.requires_grad, hybrid.parameters()),
                        lr=LR1, weight_decay=1e-4)
    sch1  = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=PHASE1_EP, eta_min=1e-5)
    best_va = 0.0; best_w = None

    for ep in range(1, PHASE1_EP + 1):
        t0 = time.time()
        tl, ta, cl, rl, kl = train_epoch(hybrid, trnl, cw, opt1, DEVICE, phase2=False)
        vl, va, _, _, _, _ = eval_epoch(hybrid, vall, cw, DEVICE)
        sch1.step()
        hist["tl"].append(tl); hist["vl"].append(vl); hist["va"].append(va)
        hist["cl"].append(cl); hist["rl"].append(rl); hist["kl"].append(kl)
        if va > best_va: best_va = va; best_w = {k:v.cpu().clone() for k,v in hybrid.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:02d}/{PHASE1_EP}  train {ta*100:6.2f}%  val {va*100:6.2f}%  lr={sch1.get_last_lr()[0]:.1e}  [{time.time()-t0:.1f}s]")

    phase1_end = PHASE1_EP

    # ---- Phase 2: unfreeze all, joint fine-tune -----------------------------
    print(f"\n{'='*60}\n  Phase 2: Joint fine-tune ({PHASE2_EP} epochs, all layers active)\n{'='*60}")
    hybrid.unfreeze_all()
    opt2  = optim.AdamW(hybrid.parameters(), lr=LR2, weight_decay=1e-4)
    sch2  = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EP, eta_min=1e-6)

    for ep in range(1, PHASE2_EP + 1):
        t0 = time.time()
        tl, ta, cl, rl, kl = train_epoch(hybrid, trnl, cw, opt2, DEVICE, phase2=True)
        vl, va, _, _, _, _ = eval_epoch(hybrid, vall, cw, DEVICE)
        sch2.step()
        hist["tl"].append(tl); hist["vl"].append(vl); hist["va"].append(va)
        hist["cl"].append(cl); hist["rl"].append(rl); hist["kl"].append(kl)
        if va > best_va: best_va = va; best_w = {k:v.cpu().clone() for k,v in hybrid.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"  Ep {ep:02d}/{PHASE2_EP}  train {ta*100:6.2f}%  val {va*100:6.2f}%  "
                  f"cls={cl:.4f}  rec={rl:.4f}  kl={kl:.4f}  lr={sch2.get_last_lr()[0]:.1e}  [{time.time()-t0:.1f}s]")

    # mark phase-1/2 boundary for plot
    hist["phase1_end"] = list(range(1, phase1_end + 1))

    hybrid.load_state_dict(best_w)
    torch.save(best_w, HYBRID_PT)
    print(f"\n  Best val acc : {best_va*100:.2f}%  -> model saved: {HYBRID_PT}")

    # ---- final test evaluation ----------------------------------------------
    _, test_acc, h_pred, h_true, h_probs, h_mus = eval_epoch(hybrid, tstl, cw, DEVICE)
    h_errors = get_recon_errors(hybrid, tstl, DEVICE)
    print(f"  Test accuracy : {test_acc*100:.2f}%")

    metrics_hyb = (accuracy_score(h_true, h_pred),
                   f1_score(h_true, h_pred, average="weighted", zero_division=0),
                   precision_score(h_true, h_pred, average="weighted", zero_division=0),
                   recall_score(h_true, h_pred, average="weighted", zero_division=0))

    metrics_dict = {"CNN": metrics_cnn, "VAE (binary)": metrics_vae, "Hybrid": metrics_hyb}

    # ---- generate all plots -------------------------------------------------
    print(f"\n{'='*60}\n  Generating publication-quality plots\n{'='*60}\n")

    plot_loss_curves(hist,
        os.path.join(REPORT_DIR, "hybrid_loss_curves.png"))
    plot_confusion_matrix(h_true, h_pred, CLASS_NAMES,
        os.path.join(REPORT_DIR, "hybrid_confusion_matrix.png"))
    plot_tsne(h_mus, h_true, CLASS_NAMES, PALETTE,
        os.path.join(REPORT_DIR, "hybrid_tsne.png"))
    plot_roc(h_true, h_probs, CLASS_NAMES, PALETTE, n_cls,
        os.path.join(REPORT_DIR, "hybrid_roc.png"))
    plot_pr(h_true, h_probs, CLASS_NAMES, PALETTE, n_cls,
        os.path.join(REPORT_DIR, "hybrid_pr.png"))
    plot_model_comparison(metrics_dict,
        os.path.join(REPORT_DIR, "model_comparison.png"))
    plot_recon_distribution(h_errors, h_true, CLASS_NAMES, PALETTE,
        os.path.join(REPORT_DIR, "hybrid_recon_dist.png"))
    plot_calibration(h_true, h_probs, CLASS_NAMES, n_cls,
        os.path.join(REPORT_DIR, "hybrid_calibration.png"))

    ha, hf = write_report(h_true, h_pred, h_probs, CLASS_NAMES, n_cls,
                           metrics_dict,
                           os.path.join(REPORT_DIR, "hybrid_metrics_report.txt"))

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  FINAL SUMMARY")
    print("=" * 68)
    for model, (ma, mf, mp, mr) in metrics_dict.items():
        print(f"  {model:<14s}  Acc {ma*100:6.2f}%   F1 {mf:.4f}   Prec {mp:.4f}   Rec {mr:.4f}")
    print("=" * 68)
    print(f"  Plots  -> {REPORT_DIR}")
    print(f"  Model  -> {HYBRID_PT}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()

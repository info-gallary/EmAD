"""
Dual anomaly detection pipeline for ESA satellite telemetry.
Model 1: 1D-CNN Classifier  (supervised, stratified split)
Model 2: Variational Autoencoder  (latent-space, trained on Normal only)
"""
import os, time, warnings
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
from sklearn.metrics import (accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
warnings.filterwarnings("ignore")

CSV_PATH   = r"d:\UbtVM-Def\Models\preprocessed_dataset.csv"
MODEL_PATH = r"d:\UbtVM-Def\Models\cnn1d_anomaly.pt"
VAE_PATH   = r"d:\UbtVM-Def\Models\vae_anomaly.pt"
REPORT_DIR = r"d:\UbtVM-Def\Models\reports"

WINDOW=50; STEP=2; BATCH=256; CNN_EPOCHS=50; VAE_EPOCHS=40
LR=3e-4; DROPOUT=0.3; LATENT_DIM=64; VAE_BETA=0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES=["Normal","Communication","Power","Thermal",
             "Software","Rare-Event","Comm-Gap","Unknown"]

# ---- dataset ----------------------------------------------------------------
def make_windows(X, y, w, s):
    Xw, yw = [], []
    for i in range(0, len(X)-w+1, s):
        seg = y[i:i+w]; v, c = np.unique(seg, return_counts=True)
        Xw.append(X[i:i+w]); yw.append(v[c.argmax()])
    return np.array(Xw, np.float32), np.array(yw, np.int64)

class TelDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.transpose(0,2,1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ---- CNN model --------------------------------------------------------------
class Res1D(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        p = k//2
        self.n = nn.Sequential(
            nn.Conv1d(ch,ch,k,padding=p,bias=False), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch,ch,k,padding=p,bias=False), nn.BatchNorm1d(ch))
        self.a = nn.GELU()
    def forward(self, x): return self.a(x + self.n(x))

class CNN1D(nn.Module):
    def __init__(self, nf, nc, dr=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(nf,64,7,padding=3,bias=False), nn.BatchNorm1d(64), nn.GELU())
        self.s1 = nn.Sequential(Res1D(64), Res1D(64))
        self.d1 = nn.Sequential(nn.Conv1d(64,128,3,stride=2,padding=1,bias=False), nn.BatchNorm1d(128), nn.GELU())
        self.s2 = nn.Sequential(Res1D(128), Res1D(128))
        self.d2 = nn.Sequential(nn.Conv1d(128,256,3,stride=2,padding=1,bias=False), nn.BatchNorm1d(256), nn.GELU())
        self.s3 = nn.Sequential(Res1D(256), Res1D(256))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256,128), nn.GELU(), nn.Dropout(dr), nn.Linear(128,nc))
    def forward(self, x):
        x = self.stem(x); x = self.d1(self.s1(x)); x = self.d2(self.s2(x)); x = self.s3(x)
        return self.head(self.pool(x))

# ---- VAE model --------------------------------------------------------------
class VAE1D(nn.Module):
    def __init__(self, nf, win, ld=64):
        super().__init__()
        self.win = win; PT = 8
        self.enc = nn.Sequential(
            nn.Conv1d(nf,128,7,padding=3,bias=False), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128,256,5,padding=2,bias=False), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(PT))
        fl = 256*PT
        self.mu_l  = nn.Linear(fl, ld)
        self.lv_l  = nn.Linear(fl, ld)
        self.dec_fc = nn.Linear(ld, fl)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (256, PT)),
            nn.ConvTranspose1d(256,128,5,stride=2,padding=2,output_padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.ConvTranspose1d(128, 64,5,stride=2,padding=2,output_padding=1), nn.BatchNorm1d(64),  nn.GELU(),
            nn.ConvTranspose1d( 64, nf,5,padding=2), nn.Sigmoid())
    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.mu_l(h), self.lv_l(h)
    def reparam(self, m, l): return m + (0.5*l).exp() * torch.randn_like(m)
    def decode(self, z): return F.interpolate(self.dec(self.dec_fc(z)), size=self.win, mode="linear", align_corners=False)
    def forward(self, x):
        m, l = self.encode(x); r = self.decode(self.reparam(m, l)); return r, m, l

def vae_loss(r, x, m, l):
    rec = F.mse_loss(r, x); kld = -0.5 * torch.mean(1 + l - m.pow(2) - l.exp())
    return rec + VAE_BETA*kld, rec.item(), kld.item()

# ---- training loops ---------------------------------------------------------
def cnn_train(model, dl, crit, opt, dev):
    model.train(); tl=tc=tn=0
    for X, y in dl:
        X,y = X.to(dev), y.to(dev); opt.zero_grad(); out=model(X); loss=crit(out,y)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl+=loss.item()*len(y); tc+=(out.argmax(1)==y).sum().item(); tn+=len(y)
    return tl/tn, tc/tn

@torch.no_grad()
def cnn_eval(model, dl, crit, dev):
    model.eval(); tl=tc=tn=0; ps,ls=[],[]
    for X, y in dl:
        X,y = X.to(dev), y.to(dev); out=model(X)
        tl+=crit(out,y).item()*len(y); p=out.argmax(1)
        tc+=(p==y).sum().item(); tn+=len(y)
        ps.extend(p.cpu().tolist()); ls.extend(y.cpu().tolist())
    return tl/tn, tc/tn, np.array(ps), np.array(ls)

def vae_train(model, dl, opt, dev):
    model.train(); tl=tr=tk=n=0
    for X, _ in dl:
        X=X.to(dev); opt.zero_grad(); r,m,l=model(X); loss,rec,kld=vae_loss(r,X,m,l)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl+=loss.item()*len(X); tr+=rec*len(X); tk+=kld*len(X); n+=len(X)
    return tl/n, tr/n, tk/n

@torch.no_grad()
def vae_errors(model, dl, dev):
    model.eval(); errs=[]
    for X, _ in dl:
        X=X.to(dev); r,_,_=model(X)
        mse=F.mse_loss(r,X,reduction="none").mean(dim=(1,2)); errs.extend(mse.cpu().tolist())
    return np.array(errs)

# ---- plots ------------------------------------------------------------------
def save_cnn_curves(h, d):
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4)); ep=range(1,len(h["tl"])+1)
    a1.plot(ep,h["tl"],label="Train"); a1.plot(ep,h["vl"],label="Val")
    a1.set(title="CNN Loss",xlabel="Epoch"); a1.legend(); a1.grid(True,alpha=0.3)
    a2.plot(ep,[x*100 for x in h["ta"]],label="Train"); a2.plot(ep,[x*100 for x in h["va"]],label="Val")
    a2.set(title="CNN Accuracy (%)",xlabel="Epoch"); a2.legend(); a2.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(d,"cnn_training_curves.png"),dpi=150); plt.close()
    print("  Saved cnn_training_curves.png")

def save_vae_curves(h, d):
    fig,ax=plt.subplots(figsize=(8,4)); ep=range(1,len(h["loss"])+1)
    ax.plot(ep,h["loss"],label="Total"); ax.plot(ep,h["rec"],label="Recon MSE"); ax.plot(ep,h["kld"],label="KL div")
    ax.set(title="VAE Training Loss",xlabel="Epoch"); ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(d,"vae_loss_curves.png"),dpi=150); plt.close()
    print("  Saved vae_loss_curves.png")

def save_cm(yt, yp, names, tag, d):
    pr=sorted(set(yt)|set(yp)); labs=[names[i] for i in pr]
    cm=confusion_matrix(yt,yp,labels=pr)
    fig,ax=plt.subplots(figsize=(max(6,len(pr)*1.5), max(5,len(pr)*1.2)))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=labs,yticklabels=labs,ax=ax)
    ax.set(xlabel="Predicted",ylabel="True",title=f"Confusion Matrix - {tag}")
    plt.tight_layout(); fname=os.path.join(d,f"cm_{tag.lower()}.png")
    plt.savefig(fname,dpi=150); plt.close(); print(f"  Saved {os.path.basename(fname)}")

def save_vae_dist(e, yt, thr, d):
    fig,ax=plt.subplots(figsize=(10,5))
    ne=e[yt==0]; ae=e[yt!=0]
    if len(ne): ax.hist(ne,bins=40,alpha=0.6,label=f"Normal (n={len(ne)})",color="green",density=True)
    if len(ae): ax.hist(ae,bins=40,alpha=0.6,label=f"Anomaly (n={len(ae)})",color="red",density=True)
    ax.axvline(thr,color="black",linestyle="--",label=f"Threshold={thr:.4f}")
    ax.set(xlabel="Reconstruction MSE",ylabel="Density",title="VAE Anomaly Score Distribution (Test)")
    ax.legend(); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(d,"vae_score_distribution.png"),dpi=150); plt.close()
    print("  Saved vae_score_distribution.png")

# ---- reports ----------------------------------------------------------------
def cnn_report(yt, yp, names, d):
    pr=sorted(set(yt)); labs=[names[i] for i in pr]
    acc=accuracy_score(yt,yp)
    f1w=f1_score(yt,yp,average="weighted",labels=pr,zero_division=0)
    f1m=f1_score(yt,yp,average="macro",labels=pr,zero_division=0)
    prec=precision_score(yt,yp,average="weighted",labels=pr,zero_division=0)
    rec=recall_score(yt,yp,average="weighted",labels=pr,zero_division=0)
    rpt=classification_report(yt,yp,labels=pr,target_names=labs,digits=4,zero_division=0)
    txt=("\n"+"="*64+"\n"
        "  ESA ANOMALY DETECTION - CNN CLASSIFIER METRICS\n"
        "  1D-CNN with Residual Blocks | Stratified 70/15/15 split\n"
        +"="*64+"\n\n"
        f"  Accuracy            : {acc*100:.2f}%\n"
        f"  Weighted F1-Score   : {f1w:.4f}\n"
        f"  Macro F1-Score      : {f1m:.4f}\n"
        f"  Weighted Precision  : {prec:.4f}\n"
        f"  Weighted Recall     : {rec:.4f}\n\n"
        +"-"*64+"\n  Per-Class Report:\n\n"+rpt+"\n")
    print(txt)
    with open(os.path.join(d,"cnn_metrics_report.txt"),"w",encoding="utf-8") as f: f.write(txt)
    print("  Saved cnn_metrics_report.txt")
    return acc, f1w

def vae_report(e, yt, thr, d):
    bp=(e>thr).astype(int); bt=(yt!=0).astype(int)
    acc=accuracy_score(bt,bp)
    f1=f1_score(bt,bp,zero_division=0)
    prec=precision_score(bt,bp,zero_division=0)
    rec=recall_score(bt,bp,zero_division=0)
    try: auc=roc_auc_score(bt,e)
    except: auc=float("nan")
    rpt=classification_report(bt,bp,target_names=["Normal","Anomaly"],digits=4,zero_division=0)
    txt=("\n"+"="*64+"\n"
        "  ESA ANOMALY DETECTION - VAE LATENT SPACE METRICS\n"
        "  VAE trained on Normal windows only | Binary detection\n"
        +"="*64+"\n\n"
        f"  Threshold (mu+2*sigma) : {thr:.6f}\n"
        f"  Accuracy (binary)      : {acc*100:.2f}%\n"
        f"  F1 (anomaly class)     : {f1:.4f}\n"
        f"  Precision (anomaly)    : {prec:.4f}\n"
        f"  Recall    (anomaly)    : {rec:.4f}\n"
        f"  ROC-AUC                : {auc:.4f}\n\n"
        +"-"*64+"\n  Binary Classification Report:\n\n"+rpt+"\n")
    print(txt)
    with open(os.path.join(d,"vae_metrics_report.txt"),"w",encoding="utf-8") as f: f.write(txt)
    print("  Saved vae_metrics_report.txt")
    return acc, f1, auc

# ---- main -------------------------------------------------------------------
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"\nDevice: {DEVICE}")

    df=pd.read_csv(CSV_PATH,index_col="timestamp")
    fc=[c for c in df.columns if c not in ("label","class_name")]
    X=df[fc].values.astype(np.float32); y=df["label"].values.astype(np.int64)
    nc=int(y.max())+1
    print(f"  Samples:{len(X):,}  Features:{X.shape[1]}  Classes:{np.unique(y).tolist()}")

    Xw,yw=make_windows(X,y,WINDOW,STEP)
    dist={int(c):int(n) for c,n in zip(*np.unique(yw,return_counts=True))}
    print(f"  Windows:{Xw.shape}  Dist:{dist}")

    n=len(Xw); n_tr,n_va=int(0.70*n),int(0.85*n)
    Xtr,ytr=Xw[:n_tr],yw[:n_tr]
    Xva,yva=Xw[n_tr:n_va],yw[n_tr:n_va]
    Xte,yte=Xw[n_va:],yw[n_va:]
    print(f"  Train:{len(Xtr):,}  Val:{len(Xva):,}  Test:{len(Xte):,}")

    ci,cn=np.unique(ytr,return_counts=True)
    wts=np.ones(nc,np.float32)
    for c,n in zip(ci,cn): wts[c]=len(ytr)/(len(ci)*n)
    cw=torch.tensor(wts,dtype=torch.float32).to(DEVICE)
    print("Class weights:",{int(c):round(float(wts[c]),3) for c in ci})

    trnl=DataLoader(TelDS(Xtr,ytr),BATCH,shuffle=True,num_workers=0)
    vall=DataLoader(TelDS(Xva,yva),BATCH,shuffle=False,num_workers=0)
    tstl=DataLoader(TelDS(Xte,yte),BATCH,shuffle=False,num_workers=0)

    # ============================ CNN ========================================
    print("\n"+"="*60+"\n  PART 1: 1D-CNN Classifier\n"+"="*60)
    cnn=CNN1D(X.shape[1],nc,DROPOUT).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in cnn.parameters() if p.requires_grad):,}")
    crit=nn.CrossEntropyLoss(weight=cw,label_smoothing=0.05)
    opt=optim.AdamW(cnn.parameters(),lr=LR,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=CNN_EPOCHS,eta_min=1e-5)
    h={"tl":[],"ta":[],"vl":[],"va":[]}; bva=0.0; bw=None
    print(f"  Training {CNN_EPOCHS} epochs...\n")
    for ep in range(1,CNN_EPOCHS+1):
        t0=time.time()
        tl,ta=cnn_train(cnn,trnl,crit,opt,DEVICE)
        vl,va,_,_=cnn_eval(cnn,vall,crit,DEVICE)
        sch.step()
        h["tl"].append(tl); h["ta"].append(ta); h["vl"].append(vl); h["va"].append(va)
        if va>bva: bva=va; bw={k:v.cpu().clone() for k,v in cnn.state_dict().items()}
        if ep%5==0 or ep==1:
            print(f"  Ep {ep:03d}/{CNN_EPOCHS}  train {ta*100:6.2f}% l={tl:.4f}  val {va*100:6.2f}% l={vl:.4f}  lr={sch.get_last_lr()[0]:.1e}  [{time.time()-t0:.1f}s]")
    cnn.load_state_dict(bw); torch.save(bw,MODEL_PATH)
    _,ca,cp,ct=cnn_eval(cnn,tstl,crit,DEVICE)
    print(f"\n  Best val: {bva*100:.2f}%   Test acc: {ca*100:.2f}%")

    # ============================ VAE ========================================
    print("\n"+"="*60+"\n  PART 2: Variational Autoencoder\n"+"="*60)
    mn=ytr==0; Xn=Xtr[mn]; yn=ytr[mn]
    if len(Xn)<10:
        print("  [WARN] Too few Normal windows - using all training data.")
        Xn,yn=Xtr,ytr
    print(f"  Normal training windows: {len(Xn)}")
    vae=VAE1D(X.shape[1],WINDOW,LATENT_DIM).to(DEVICE)
    vo=optim.Adam(vae.parameters(),lr=LR)
    vs=optim.lr_scheduler.CosineAnnealingLR(vo,T_max=VAE_EPOCHS,eta_min=1e-5)
    print(f"  Params: {sum(p.numel() for p in vae.parameters() if p.requires_grad):,}")
    ndl=DataLoader(TelDS(Xn,yn),BATCH,shuffle=True,num_workers=0)
    vh={"loss":[],"rec":[],"kld":[]}; bvl=float("inf"); bvw=None
    print(f"  Training {VAE_EPOCHS} epochs...\n")
    for ep in range(1,VAE_EPOCHS+1):
        t0=time.time(); tl,tr,tk=vae_train(vae,ndl,vo,DEVICE); vs.step()
        vh["loss"].append(tl); vh["rec"].append(tr); vh["kld"].append(tk)
        if tl<bvl: bvl=tl; bvw={k:v.cpu().clone() for k,v in vae.state_dict().items()}
        if ep%5==0 or ep==1:
            print(f"  Ep {ep:03d}/{VAE_EPOCHS}  loss={tl:.4f}  recon={tr:.4f}  kld={tk:.4f}  [{time.time()-t0:.1f}s]")
    vae.load_state_dict(bvw); torch.save(bvw,VAE_PATH)
    en=vae_errors(vae,DataLoader(TelDS(Xn,yn),BATCH,num_workers=0),DEVICE)
    thr=float(np.mean(en)+2*np.std(en))
    et=vae_errors(vae,tstl,DEVICE)
    print(f"\n  Threshold: {thr:.6f}   VAE saved -> {VAE_PATH}")

    # ============================ Reports =====================================
    print("\n"+"="*60+"\n  PART 3: Saving reports\n"+"="*60+"\n")
    save_cnn_curves(h,REPORT_DIR)
    save_vae_curves(vh,REPORT_DIR)
    save_cm(ct,cp,CLASS_NAMES,"CNN",REPORT_DIR)
    save_cm((yte!=0).astype(int),(et>thr).astype(int),["Normal","Anomaly"],"VAE",REPORT_DIR)
    save_vae_dist(et,yte,thr,REPORT_DIR)
    ca2,cf1=cnn_report(ct,cp,CLASS_NAMES,REPORT_DIR)
    va2,vf1,vauc=vae_report(et,yte,thr,REPORT_DIR)

    print("\n"+"="*64)
    print("  COMBINED SUMMARY")
    print("="*64)
    print(f"  [CNN] Accuracy {ca2*100:.2f}%   Weighted-F1 {cf1:.4f}")
    print(f"  [VAE] Binary Acc {va2*100:.2f}%   F1 {vf1:.4f}   ROC-AUC {vauc:.4f}")
    print(f"  All reports -> {REPORT_DIR}")
    print("="*64+"\n")

if __name__ == "__main__":
    main()

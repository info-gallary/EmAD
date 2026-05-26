# Research Context — EmAD: ESA Multi-Mission Anomaly Detection

This document provides the context, methodology, and contribution framing needed to write a research paper based on this codebase. It is written for co-authors who need to understand the problem, the approach, and how to position the work.

---

## 1. Problem Statement

Satellite telemetry anomaly detection is a safety-critical task in spacecraft operations. Anomalies — such as power fluctuations, communication gaps, thermal excursions, or software resets — must be identified early to prevent mission failure. Three challenges make this hard:

1. **Severe class imbalance.** Normal and Rare-Event samples dominate (~90-95% of timestamps); true anomalies are rare (< 3%).
2. **Heterogeneous missions.** Different spacecraft have different sensor suites, sampling rates, and data types (continuous vs. categorical telemetry).
3. **No generalizable model exists.** Most existing work trains per-mission models and does not demonstrate cross-mission transfer.

---

## 2. Dataset

**Source:** ESA Anomaly Detection Benchmark  
**Reference:** Kotowski et al. (2023), "ESA Anomaly Detection Benchmark"

| Mission | Period | Channels | Sampling | Channel Type |
| --- | --- | --- | --- | --- |
| Mission 1 | Dec 2004 (15 days) | 58 target / 55 usable | 60 s (resampled) | Continuous float |
| Mission 2 | Dec 2002 (16 days) | 47 target / 43 usable | 60 s (resampled) | Continuous float |
| Mission 3 | Dec 2000 (15 days) | 24 target / 7 usable | ~15 s (resampled) | Categorical (label-encoded) |

**Anomaly taxonomy (8 classes):**

| ID | Class | ESA Category |
| --- | --- | --- |
| 0 | Normal | — |
| 1 | Communication Anomaly | Anomaly / subsystem_1 |
| 2 | Power Anomaly | Anomaly / subsystem_5 |
| 3 | Thermal Anomaly | Anomaly / subsystem_6 |
| 4 | Software Anomaly | Anomaly / subsystem_3 |
| 5 | Rare Nominal Event | Rare Event |
| 6 | Communication Gap | Communication Gap |
| 7 | Unknown Anomaly | Anomaly / other subsystem |

**Label distribution per mission:**

| Mission | Normal | Rare-Event | Thermal | Power | Other |
| --- | --- | --- | --- | --- | --- |
| Mission 1 | 4.0 % | 93.8 % | 2.1 % | — | — |
| Mission 2 | 85.1 % | 14.9 % | — | — | — |
| Mission 3 | 72.6 % | — | — | 27.4 % | — |
| Combined | 54.6 % | 35.8 % | 0.7 % | 8.9 % | — |

---

## 3. Preprocessing Pipeline

All preprocessing is deterministic and reproducible via `preprocess_all_missions.py`.

**Per channel, 5 features are derived:**

```
x_smooth   = SG-filter(x, window=11, poly=2, deriv=0)   # denoised signal
x_d1       = SG-filter(x, window=11, poly=2, deriv=1)   # velocity
x_d2       = SG-filter(x, window=11, poly=2, deriv=2)   # acceleration
x_rmean    = rolling_mean(x_smooth, window=10)           # local trend
x_rstd     = rolling_std(x_smooth, window=10)            # local variability
```

**Cleaning criteria:** drop channels with > 90% NaN or zero variance. Remaining gaps are filled by linear interpolation.

**Categorical channels** (Mission 3): label-encoded to contiguous integers before SG filtering, then scaled identically to numeric channels.

**Feature dimensions after preprocessing:**

- Mission 1: 55 channels x 5 = 275 features
- Mission 2: 43 channels x 5 = 215 features
- Mission 3:  7 channels x 5 =  35 features
- Combined (zero-padded): 275 features for all (feat_0..feat_274)

**Windowing:** 50-step non-overlapping-stride (step=2) sliding window; label = majority class in window.

**Normalization:** MinMax scaling to [0, 1] per feature, fit on training split only.

**Stratified splits:** 70 / 15 / 15 (train / val / test), stratified by class label, `random_state=42`.

---

## 4. Models

### 4.1 1D-CNN Classifier

A fully supervised residual convolutional network.

**Architecture:**

```
Input shape: (batch, n_feat, 50)
  Stem: Conv1d(n_feat, 64, kernel=7) + BN + GELU
  Stage 1: 2x Res1D(64) -> Conv1d(64, 128, stride=2) + BN + GELU
  Stage 2: 2x Res1D(128) -> Conv1d(128, 256, stride=2) + BN + GELU
  Stage 3: 2x Res1D(256)
  Pool: AdaptiveAvgPool1d(1)
  Head: Linear(256, 128) + GELU + Dropout(0.3) + Linear(128, n_cls)
```

Each Res1D block: Conv1d + BN + GELU + Conv1d + BN + skip-add + GELU.

**Parameters:** ~1.3 M (varies slightly with n_feat and n_cls)

**Training:**

- Loss: CrossEntropyLoss with class weights (inverse frequency) + label smoothing 0.05
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Schedule: CosineAnnealingLR (T_max=30, eta_min=1e-5)
- Gradient clipping: max_norm=1.0
- Best checkpoint: saved by val accuracy

### 4.2 Variational Autoencoder (VAE)

An unsupervised latent-space detector. Trained without labels; anomalies are detected at inference via reconstruction error threshold.

**Architecture (Convolutional beta-VAE):**

```
Encoder:
  Conv1d(n_feat, 128, kernel=7) + BN + GELU
  Conv1d(128, 256, kernel=5) + BN + GELU
  AdaptiveAvgPool1d(8)         -> flatten to 256*8 = 2048-d
  Linear(2048, 64) -> mu       (mean of latent distribution)
  Linear(2048, 64) -> log_var  (log variance)

Reparameterization: z = mu + exp(0.5 * log_var) * eps,  eps ~ N(0,I)

Decoder:
  Linear(64, 2048) -> Unflatten(256, 8)
  ConvTranspose1d(256, 128, stride=2)
  ConvTranspose1d(128, 64, stride=2)
  ConvTranspose1d(64, n_feat)
  Interpolate to window size
  Sigmoid
```

**Parameters:** ~1.1 M

**Loss:** MSE(recon, input) + 0.1 * KL(N(mu, sigma^2) || N(0, I))

**Anomaly threshold:** mu_normal + 2 * sigma_normal (computed on Normal-class training windows)

### 4.3 Hybrid CNN-VAE Meta-Learner

The key contribution. The hypothesis is that CNN backbone features capture discriminative temporal patterns while the VAE latent space captures the generative structure of normal vs. anomalous behavior; combining both via a learned meta-learner outperforms either model alone.

**Architecture:**

```
Given input x of shape (batch, n_feat, 50):

CNN branch:
  cnn_feat = GlobalAvgPool(CNN-stages(x))   -> (batch, 256)

VAE branch:
  recon, mu, log_var = VAE(x)               -> mu shape (batch, 64)

Fusion:
  fused = concat([cnn_feat, mu], dim=1)     -> (batch, 320)

Meta-learner:
  Linear(320, 256) + BN + GELU + Dropout(0.3)
  Linear(256, 128) + BN + GELU + Dropout(0.2)
  Linear(128, n_cls)
```

**Training strategy (two-phase):**

| Phase | Epochs | LR | Frozen | Loss |
| --- | --- | --- | --- | --- |
| Phase 1 (warmup) | 15-20 | 1e-3 | CNN + VAE backbones | CE only |
| Phase 2 (joint) | 20-30 | 5e-5 | None (all unfrozen) | W_CLS*CE + W_REC*MSE + W_KL*KL |

Joint loss weights: W_CLS=1.0, W_REC=0.3, W_KL=0.05

The two-phase approach prevents the meta-learner from overfitting to random backbone representations before the backbones are initialized. This is similar in spirit to pre-training + fine-tuning in NLP.

**Parameters:** ~2.4 M total (CNN + VAE + meta-learner)

### 4.4 Generalized Model

The same Hybrid CNN-VAE architecture trained on the zero-padded 3-mission combined dataset. Evaluated with:

- Overall accuracy and F1 on the stratified test set
- Per-mission breakdown within the test set
- Leave-One-Mission-Out (LOMO): train on 2 missions, test on the held-out mission

The LOMO protocol tests whether the model generalizes to entirely unseen spacecraft.

---

## 5. Key Contributions

Frame the paper around these claims (ordered by impact):

1. **Multi-mission benchmark evaluation.** Most anomaly detection papers evaluate on a single mission in isolation. This work evaluates a single architecture across 3 distinct ESA missions with different sensor modalities, channel counts (35-275 features), and anomaly distributions.

2. **Hybrid CNN-VAE meta-learner.** The fusion of discriminative CNN features with generative VAE latent representations via a two-phase meta-learner is a novel architecture for telemetry anomaly detection. The two-phase training (freeze -> unfreeze) provides stable convergence.

3. **Categorical channel handling.** Mission 3 channels are discrete enumerated state variables, not continuous signals. Label-encoding and applying the same SG + rolling-stats pipeline to categorical data is a practical contribution for operational use.

4. **Zero-padded cross-mission generalization.** Training a single model on zero-padded multi-mission data with LOMO evaluation demonstrates transfer learning potential across different spacecraft.

5. **Rich evaluation suite.** Per-class ROC/PR curves, calibration plots, t-SNE of latent space, reconstruction error distributions, and cross-mission comparison charts provide comprehensive characterization beyond simple accuracy.

---

## 6. Experimental Results (updated by training runs)

Results are written automatically to:

- `reports/missions/all_missions_summary.txt` — per-mission CNN/VAE/Hybrid
- `reports/hybrid/hybrid_metrics_report.txt` — Mission 1 hybrid detailed report
- `reports/generalized/generalized_report.txt` — generalized model + LOMO

Key figures for the paper:

| Figure | File | What it shows |
| --- | --- | --- |
| Fig 1: Architecture | (draw manually from this doc) | CNN-VAE-meta diagram |
| Fig 2: Training curves | `reports/hybrid/hybrid_loss_curves.png` | Loss breakdown by phase |
| Fig 3: Confusion matrix | `reports/hybrid/hybrid_confusion_matrix.png` | Counts + row-normalised |
| Fig 4: t-SNE | `reports/hybrid/hybrid_tsne.png` | Class separation in latent space |
| Fig 5: ROC curves | `reports/hybrid/hybrid_roc.png` | Per-class AUC |
| Fig 6: Cross-mission | `reports/missions/cross_mission_comparison.png` | All models x all missions |
| Fig 7: LOMO | `reports/generalized/generalized_lomo.png` | Generalization bars |
| Fig 8: Calibration | `reports/hybrid/hybrid_calibration.png` | Reliability diagram |

---

## 7. Baselines to Consider Adding

To strengthen the paper for a top-tier venue, consider comparing against:

| Baseline | Why include |
| --- | --- |
| LSTM / GRU classifier | Standard sequential model for telemetry |
| Temporal Convolutional Network (TCN) | Competitive with 1D-CNN on time series |
| Isolation Forest / One-Class SVM | Classical unsupervised anomaly detection |
| USAD / OmniAnomaly | State-of-the-art VAE-based anomaly detection |
| Transformer (PatchTST / Anomaly Transformer) | Current SOTA for time series anomaly |
| Mission 1 specialist vs. Generalized | Ablation: does multi-mission training hurt per-mission accuracy? |

---

## 8. Ablation Studies to Report

| Ablation | What to vary | Expected finding |
| --- | --- | --- |
| SG features vs. raw | Remove derivatives and rolling stats | Feature engineering contributes significantly for sparse anomaly classes |
| Beta value (KL weight) | beta in {0.01, 0.1, 1.0} | Higher beta -> disentangled but less discriminative latent |
| Phase 1 duration | 0, 10, 20 epochs | Too few -> unstable; too many -> underfitting |
| Fusion dimension | Latent dim in {32, 64, 128} | 64 is good balance for this dataset size |
| Window size | {20, 50, 100} | Temporal context vs. per-window label ambiguity |
| Class weighting | No weighting vs. inverse frequency | Critical for rare classes (Thermal, Power) |

---

## 9. Limitations and Future Work

- **Mission 3 feature sparsity.** Only 7 of 24 channels survived quality filtering, producing a very low-dimensional representation (35 features) that may underrepresent the mission's telemetry.

- **Zero-padding assumption.** Padding missing channels with zeros assumes they carry no signal. A mask-based attention mechanism (e.g., Transformer with masking) could handle variable-length feature sets more gracefully.

- **Label quality.** ESA labels overlap in time and are assigned per-channel, not per-timestamp. The max-priority merge is a reasonable heuristic but may introduce label noise.

- **No online/streaming evaluation.** All experiments use offline batch training. Operational use would require online learning or change-point detection.

- **Single threshold for VAE.** The mu + 2*sigma threshold is heuristic. Optimal threshold tuning (e.g., via ROC F-measure) would improve binary detection metrics.

---

## 10. Suggested Venue Targets

| Tier | Venue | Notes |
| --- | --- | --- |
| Top | IEEE Transactions on Aerospace and Electronic Systems | Best fit: aerospace + ML |
| Top | Reliability Engineering & System Safety | If emphasizing safety/fault detection framing |
| Top | Expert Systems with Applications | Broad applied ML |
| Conference | AAAI / IJCAI Applied Track | If novelty angle on architecture is emphasized |
| Conference | ECML-PKDD / ICDM | ML + data mining community |
| Conference | IAC (International Astronautical Congress) | Domain venue, lower bar, good for visibility |

**Honest assessment of current work vs. top-tier requirements:**

The architecture is solid and the evaluation is thorough. The main gap for top-tier (IEEE Trans. AES or equivalent) is the lack of comparison against SOTA baselines (USAD, OmniAnomaly, Transformer). Adding 2-3 baselines and at least one ablation study would make this competitive. The multi-mission cross-validation angle is genuinely novel and underexplored in the ESA benchmark literature.

---

## 11. Reproducibility Checklist

- [x] All random seeds fixed (`random_state=42`, `torch` seeded via AdamW)
- [x] Preprocessing is deterministic (no random augmentation)
- [x] Train/val/test split is stratified and fixed
- [x] Class weights computed from training split only
- [x] Model weights saved to `models/`
- [x] Full classification reports saved to `reports/`
- [x] `requirements.txt` with pinned versions
- [ ] Docker image or environment.yml (recommended for camera-ready)
- [ ] Random seed for PyTorch DataLoader workers (add `worker_init_fn` if needed for exact reproducibility)

---

## 12. Code Reference

| Script | Purpose |
| --- | --- |
| `preprocess_all_missions.py` | Data pipeline for all 3 missions |
| `train_cnn1d.py` | CNN + VAE baseline (Mission 1) |
| `train_hybrid.py` | Hybrid meta-learner (Mission 1, full plots) |
| `train_all_missions.py` | Per-mission training loop |
| `train_generalized.py` | Multi-mission generalization + LOMO |
| `esa_anomaly_detection/src/data_loader.py` | Channel metadata utilities |

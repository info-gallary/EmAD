# EmAD — ESA Multi-Mission Anomaly Detection

Deep learning system for multiclass anomaly detection across **3 ESA satellite telemetry missions** using a 3-model ensemble pipeline: 1D-CNN, Variational Autoencoder (VAE), and a Hybrid CNN-VAE Meta-Learner.

## Performance Summary

### Mission 1 (Dec 2004 — 55 channels, 275 features)

| Model | Test Accuracy | Weighted F1 |
|---|---|---|
| 1D-CNN | 99.88 % | 0.9988 |
| VAE (binary) | 92.95 % | 0.9621 (AUC 0.9669) |
| **Hybrid Meta-Learner** | **99.81 %** | **0.9982** |

### Multi-Mission Results (per-mission training)

| Mission | Features | Classes | CNN Acc | Hybrid Acc | Hybrid W-F1 |
| --- | --- | --- | --- | --- | --- |
| Mission 1 | 275 | Normal, Thermal, Rare-Event | 99.87 % | 99.80 % | 0.9980 |
| Mission 2 | 215 | Normal, Rare-Event | 99.63 % | 99.63 % | 0.9963 |
| Mission 3 | 35 | Normal, Power | 100.00 % | 100.00 % | 1.0000 |

### Generalized Model (trained on all 3 missions)

| Metric | Value |
| --- | --- |
| Overall Test Accuracy | 97.69 % |
| Weighted F1 | 0.9837 |
| Macro F1 | 0.8395 |
| Mission 1 (test subset) | 99.60 % |
| Mission 2 (test subset) | 93.97 % |
| Mission 3 (test subset) | 99.74 % |

> Full LOMO generalization results in `reports/generalized/generalized_report.txt`.

---

## Anomaly Classes

| ID | Class |
|---|---|
| 0 | Normal |
| 1 | Communication Anomaly |
| 2 | Power / Electrical Anomaly |
| 3 | Thermal Anomaly |
| 4 | Software / Reset / Computer Anomaly |
| 5 | Rare Nominal Event |
| 6 | Communication Gap |
| 7 | Unknown Anomaly |

---

## Project Structure

```
EmAD/
|
|-- preprocess_to_csv.py          # Mission 1 preprocessing -> preprocessed_dataset.csv
|-- preprocess_all_missions.py    # All 3 missions -> data/missionN_preprocessed.csv
|-- train_cnn1d.py                # CNN + VAE on Mission 1 (original)
|-- train_hybrid.py               # Hybrid CNN-VAE meta-learner on Mission 1
|-- train_all_missions.py         # CNN + VAE + Hybrid on each of 3 missions
|-- train_generalized.py          # Universal model on combined 3-mission dataset
|
|-- data/
|   |-- mission1_preprocessed.csv # 20,160 x 275 features  [Git LFS]
|   |-- mission2_preprocessed.csv # 21,600 x 215 features  [Git LFS]
|   |-- mission3_preprocessed.csv # 20,160 x  35 features  [Git LFS]
|   `-- all_missions_combined.csv # 61,920 x 275 features (zero-padded)  [Git LFS]
|
|-- models/
|   |-- mN_cnn.pt                 # Per-mission CNN weights
|   |-- mN_vae.pt                 # Per-mission VAE weights
|   |-- mN_hybrid.pt              # Per-mission Hybrid weights
|   `-- generalized_hybrid.pt     # Universal cross-mission model
|
|-- reports/
|   |-- hybrid/                   # Mission 1 hybrid plots + metrics
|   |-- missions/
|   |   |-- mN/                   # Per-mission plots + metrics
|   |   |-- cross_mission_comparison.png
|   |   `-- all_missions_summary.txt
|   `-- generalized/              # Generalized model plots + LOMO results
|
|-- preprocessed_dataset.csv      # Mission 1 only (legacy)  [Git LFS]
|-- cnn1d_anomaly.pt              # Mission 1 CNN
|-- vae_anomaly.pt                # Mission 1 VAE
|-- hybrid_anomaly.pt             # Mission 1 Hybrid
|-- requirements.txt
`-- RESEARCH_CONTEXT.md           # For co-authors writing the research paper
```

> Raw data (`ESA-data/`, ~30 GB) is not included. Download from the ESA Anomaly Detection Benchmark.

---

## Preprocessing Pipeline

### Single-Mission (`preprocess_to_csv.py`)

Processes ESA-Mission1 telemetry into a flat tabular CSV:

1. Load **55 target channels** from `ESA-Mission1`, resample to 60 s
2. **Clean** — drop channels with > 90 % NaN or zero variance; linear-interpolate gaps
3. **Savitzky-Golay filter** (window = 11, poly = 2) — smooth + 1st and 2nd derivatives
4. **Rolling statistics** — 10-sample rolling mean and std per channel
5. **Label assignment** — per-timestamp multiclass label from `labels.csv` (max-priority merge)
6. **MinMax scaling** to [0, 1]
7. Save: `preprocessed_dataset.csv` (21,600 rows x 277 columns)

### Multi-Mission (`preprocess_all_missions.py`)

Handles all 3 missions. Mission 3 has categorical (enumerated) channels — these are label-encoded before processing. Output:

```
data/mission1_preprocessed.csv   275 features  (55 channels x 5)
data/mission2_preprocessed.csv   215 features  (43 channels x 5)
data/mission3_preprocessed.csv    35 features  ( 7 channels x 5)
data/all_missions_combined.csv   275 features  (zero-padded, feat_0..feat_274)
```

Feature engineering per channel:

```
channel_N             SG-smoothed telemetry value
channel_N_d1          1st derivative (velocity)
channel_N_d2          2nd derivative (acceleration)
channel_N_rmean       10-sample rolling mean
channel_N_rstd        10-sample rolling std
```

---

## Model Architecture

### 1D-CNN Classifier

Residual convolutional network over 50-step sliding windows.

```
Input (n_feat, 50)
  Stem Conv7 -> 64 ch
  2x ResBlock(64) -> stride-2 Conv -> 128 ch
  2x ResBlock(128) -> stride-2 Conv -> 256 ch
  2x ResBlock(256) -> AdaptiveAvgPool(1)
  Linear(256->128) -> Dropout(0.3) -> Linear(128->n_cls)
```

- Loss: Weighted CrossEntropy + label smoothing 0.05
- Optimizer: AdamW, lr = 1e-3, weight decay = 1e-4
- Schedule: CosineAnnealingLR

### Variational Autoencoder (VAE)

Convolutional VAE for latent-space anomaly scoring.

```
Encoder: Conv1D(7) -> Conv1D(5) -> AdaptPool(8) -> Linear -> mu, log_var
Latent:  z ~ N(mu, sigma^2),  dim = 64
Decoder: Linear -> ConvTranspose1D x 3 -> Interpolate -> Sigmoid
```

- Loss: MSE reconstruction + beta-KL (beta = 0.1)
- Inference: reconstruction MSE > (mu_normal + 2*sigma_normal) flags anomaly

### Hybrid CNN-VAE Meta-Learner

Two-phase training that fuses CNN backbone features with VAE latent representations.

```
Input (n_feat, 50)
    |                         |
CNN backbone              VAE encoder
stem->s1->d1->s2->d2->s3  enc->AdaptPool->mu
    |                         |
 GlobalAvgPool (256-d)    Latent mu (64-d)
    |                         |
    +--------- concat ---------+  (320-d)
                    |
               Meta-Learner MLP
               320 -> 256 -> 128 -> n_cls
```

**Training strategy:**

- Phase 1 (15-20 ep, lr = 1e-3): freeze CNN + VAE, train meta-learner only
- Phase 2 (20-30 ep, lr = 5e-5): unfreeze all, joint loss = W_CLS*CE + W_REC*MSE + W_KL*KL

### Generalized Model (`train_generalized.py`)

Single hybrid model trained on all 3 missions simultaneously using the zero-padded 275-feature combined dataset. Evaluated with:

- Overall test accuracy
- Per-mission breakdown
- Leave-One-Mission-Out (LOMO) generalization test

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place ESA data

```
ESA-data/
|-- ESA-Mission1/ESA-Mission1/
|-- ESA-Mission2/ESA-Mission2/
`-- ESA-Mission3/ESA-Mission3/
    |-- anomaly_types.csv
    |-- channels.csv
    |-- labels.csv
    `-- channels/channel_N/channel_N
```

### 3. Preprocess all missions

```bash
python preprocess_all_missions.py
# Output: data/missionN_preprocessed.csv, data/all_missions_combined.csv
```

### 4. Train per-mission models

```bash
python train_all_missions.py
# Output: models/mN_*.pt, reports/missions/
```

### 5. Train generalized model

```bash
python train_generalized.py
# Output: models/generalized_hybrid.pt, reports/generalized/
```

### Legacy (Mission 1 only)

```bash
python preprocess_to_csv.py
python train_cnn1d.py       # CNN + VAE
python train_hybrid.py      # Hybrid meta-learner
```

---

## Publication-Quality Plots

Each training run produces the following plots (300 DPI, publication-ready):

| File | Description |
|---|---|
| `*_training_curve.png` | Train loss and val accuracy over epochs |
| `*_confusion_matrix.png` | Counts and row-normalised confusion matrix |
| `*_roc.png` | Per-class ROC with AUC (one-vs-rest) |
| `hybrid_tsne.png` | t-SNE of VAE latent space, coloured by class |
| `hybrid_pr.png` | Per-class Precision-Recall with average precision |
| `model_comparison.png` | Grouped bar chart: CNN vs VAE vs Hybrid |
| `hybrid_recon_dist.png` | VAE reconstruction error distribution by class |
| `hybrid_calibration.png` | Reliability diagram (calibration) |
| `cross_mission_comparison.png` | Cross-mission bars for all 4 metrics |
| `generalized_lomo.png` | Leave-One-Mission-Out generalization bars |

---

## Configuration

Key hyperparameters (edit at the top of each script):

| Parameter | Value | Description |
|---|---|---|
| `WINDOW` | 50 | Sliding window length (time steps) |
| `STEP` | 2 | Window stride |
| `LATENT_DIM` | 64 | VAE latent space dimension |
| `BATCH` | 256 | Mini-batch size |
| `PHASE1_EP` | 15-20 | Meta-learner warmup epochs |
| `PHASE2_EP` | 20-30 | Joint fine-tuning epochs |
| `W_CLS` | 1.0 | Classification loss weight |
| `W_REC` | 0.3 | Reconstruction loss weight |
| `W_KL` | 0.05 | KL divergence weight |

---

## Data

CSV datasets are tracked via **Git LFS**. The combined dataset uses generic column names (`feat_0`..`feat_274`) so all missions share a consistent feature space.

| Dataset | Rows | Features | Missions |
| --- | --- | --- | --- |
| `preprocessed_dataset.csv` | 21,600 | 275 | Mission 1 only |
| `data/mission1_preprocessed.csv` | 20,160 | 275 | Mission 1 |
| `data/mission2_preprocessed.csv` | 21,600 | 215 | Mission 2 |
| `data/mission3_preprocessed.csv` | 20,160 | 35 | Mission 3 |
| `data/all_missions_combined.csv` | 61,920 | 275 | All 3 (zero-padded) |

---

## License

Research and educational use only. ESA telemetry data is subject to ESA's data usage terms.

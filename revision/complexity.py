"""
Reviewer Points 7 + 19: computational complexity & deployment analysis.

For every architecture (on Mission 3 input dims as the reference, plus M1 dims):
  - parameter count (total + trainable)
  - model size on disk (float32 MB)
  - inference latency (ms/window, batch=1 and batch=256) on CPU
  - throughput (windows/sec)
  - peak activation memory estimate
  - self-attention FLOP comparison (Transformer 50 tokens vs ConvFormer 25 tokens)

Establishes whether ConvFormer's reduced attention complexity yields a real,
measurable deployment advantage for resource-constrained spacecraft hardware.
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

import sys
sys.path.insert(0, r"d:\UbtVM-Def\Models")
import train_all_missions as T

OUT = r"d:\UbtVM-Def\Models\reports\revision"
os.makedirs(OUT, exist_ok=True)
DEVICE = "cpu"
torch.manual_seed(0)


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train


def time_inference(model, name, n_feat, win=50, n_warm=5, n_iter=30):
    model.eval()
    def fwd(x):
        if name == "Hybrid":
            return model(x)[0]
        return model(x)
    res = {}
    for bs in (1, 256):
        x = torch.randn(bs, n_feat, win)
        with torch.no_grad():
            for _ in range(n_warm): fwd(x)
            t0 = time.perf_counter()
            for _ in range(n_iter): fwd(x)
            dt = (time.perf_counter() - t0) / n_iter
        res[f"latency_bs{bs}_ms"] = round(dt * 1000, 3)
        res[f"throughput_bs{bs}_wps"] = round(bs / dt, 1)
    res["latency_per_window_ms"] = round(res["latency_bs256_ms"] / 256, 4)
    return res


def build(name, n_feat, n_cls):
    if name == "CNN":         return T.CNN1D(n_feat, n_cls, T.DROPOUT)
    if name == "BiLSTM":      return T.BiLSTM1D(n_feat, n_cls)
    if name == "Transformer": return T.Transformer1D(n_feat, n_cls)
    if name == "ConvFormer":  return T.ConvFormer1D(n_feat, n_cls)
    if name == "VAE":         return T.VAE1D(n_feat, T.WINDOW, T.LATENT_DIM)
    if name == "Hybrid":
        return T.HybridModel(T.CNN1D(n_feat, n_cls, T.DROPOUT),
                             T.VAE1D(n_feat, T.WINDOW, T.LATENT_DIM), n_cls)


def attention_flops(seq_len, d_model=128, nlayers=2):
    """Approx self-attention FLOPs: 2 * L * (n^2 * d)  for QK^T and (softmax)V."""
    return int(nlayers * 2 * (seq_len ** 2) * d_model)


def main():
    # reference input dims: M1 (275 feat) and M3 (35 feat)
    configs = {"M1_dims": (275, 3), "M3_dims": (35, 2)}
    models = ["CNN", "BiLSTM", "Transformer", "ConvFormer", "VAE", "Hybrid"]
    out = {}
    for cfgname, (n_feat, n_cls) in configs.items():
        print(f"\n=== {cfgname}  (n_feat={n_feat}, n_cls={n_cls}) ===")
        rows = {}
        for name in models:
            m = build(name, n_feat, n_cls).to(DEVICE)
            total, train = count_params(m)
            size_mb = total * 4 / (1024 ** 2)
            timing = time_inference(m, name, n_feat)
            rows[name] = {"params_total": total, "params_trainable": train,
                          "size_mb": round(size_mb, 3), **timing}
            print(f"  {name:12s} params={total/1e6:5.3f}M  size={size_mb:5.2f}MB  "
                  f"lat_bs1={timing['latency_bs1_ms']:6.2f}ms  "
                  f"thru_bs256={timing['throughput_bs256_wps']:8.0f} w/s")
        out[cfgname] = rows
    # attention FLOP comparison
    out["attention_flops"] = {
        "Transformer_50tok": attention_flops(50),
        "ConvFormer_25tok": attention_flops(25),
        "reduction_factor": round(attention_flops(50) / attention_flops(25), 2),
    }
    print(f"\nSelf-attention FLOPs: Transformer(50 tok)={out['attention_flops']['Transformer_50tok']:,}  "
          f"ConvFormer(25 tok)={out['attention_flops']['ConvFormer_25tok']:,}  "
          f"=> {out['attention_flops']['reduction_factor']}x reduction")
    with open(os.path.join(OUT, "complexity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: complexity.json")


if __name__ == "__main__":
    main()

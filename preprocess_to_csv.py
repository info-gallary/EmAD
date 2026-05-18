"""
Preprocessing pipeline: ESA Mission 1 telemetry → preprocessed_dataset.csv

Steps
-----
1. Load target channels, resample to 60-second intervals
2. Clean: drop high-NaN / constant channels, linear-interpolate gaps
3. Savitzky-Golay smooth + extract 1st & 2nd derivatives per channel
4. Add rolling mean and std (window=10) per channel
5. Assign per-timestamp multiclass label (0-7) from labels.csv
6. MinMax-scale all feature columns to [0, 1]
7. Save to preprocessed_dataset.csv

Classes
-------
  0  Normal
  1  Communication Anomaly   (subsystem_1)
  2  Power/Electrical Anomaly (subsystem_5)
  3  Thermal Anomaly          (subsystem_6)
  4  Software Anomaly         (subsystem_3)
  5  Rare Nominal Event
  6  Communication Gap
  7  Unknown Anomaly
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ── configuration ─────────────────────────────────────────────────────────────
BASE_DIR    = r"d:\UbtVM-Def\Models\ESA-data\ESA-Mission1\ESA-Mission1"
OUTPUT_CSV  = r"d:\UbtVM-Def\Models\preprocessed_dataset.csv"

DATE_START    = "2004-12-01"
DATE_END      = "2004-12-15"
RESAMPLE_FREQ = "60s"

SG_WINDOW    = 11   # must be odd, >= polyorder+2
SG_POLYORDER = 2
ROLLING_WIN  = 10

SUBSYSTEM_MAP = {
    "subsystem_1": "communication",
    "subsystem_5": "power",
    "subsystem_6": "thermal",
    "subsystem_3": "software",
}

CLASS_NAMES = {
    0: "Normal",
    1: "Communication Anomaly",
    2: "Power/Electrical Anomaly",
    3: "Thermal Anomaly",
    4: "Software Anomaly",
    5: "Rare Nominal Event",
    6: "Communication Gap",
    7: "Unknown Anomaly",
}


# ── metadata ──────────────────────────────────────────────────────────────────

def load_metadata():
    types_df    = pd.read_csv(os.path.join(BASE_DIR, "anomaly_types.csv"))
    channels_df = pd.read_csv(os.path.join(BASE_DIR, "channels.csv"))
    labels_df   = pd.read_csv(os.path.join(BASE_DIR, "labels.csv"))

    # merge labels → anomaly type → channel subsystem
    merged = (labels_df
              .merge(types_df,                       on="ID")
              .merge(channels_df[["Channel", "Subsystem"]], on="Channel"))

    def assign_class(row):
        cat = row.get("Category", "")
        sub = row.get("Subsystem", "")
        if cat == "Rare Event":
            return 5
        if cat == "Communication Gap":
            return 6
        if cat == "Anomaly":
            sem = SUBSYSTEM_MAP.get(sub, "unknown")
            return {"communication": 1, "power": 2,
                    "thermal": 3,       "software": 4}.get(sem, 7)
        return 7

    merged["class_id"] = merged.apply(assign_class, axis=1)
    return merged, channels_df


# ── channel loading ───────────────────────────────────────────────────────────

def load_target_channels(channels_df):
    targets = channels_df[channels_df["Target"].str.strip().str.upper() == "YES"]["Channel"].tolist()
    print(f"  Target channels found: {len(targets)}")

    frames = []
    for ch in tqdm(targets, desc="  Loading"):
        path = os.path.join(BASE_DIR, "channels", ch, ch)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_pickle(path).astype("float32")
            # strip timezone if present
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index().loc[DATE_START:DATE_END]
            if df.empty:
                continue
            df = df.resample(RESAMPLE_FREQ).mean()
            df.columns = [ch]
            frames.append(df)
        except Exception as exc:
            print(f"    [WARN] {ch}: {exc}")

    if not frames:
        raise RuntimeError("No target channels loaded — check BASE_DIR path.")
    return pd.concat(frames, axis=1)


# ── cleaning ──────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df.columns)
    # drop channels with > 90 % NaN
    df = df.loc[:, df.isna().mean() < 0.90]
    # drop constant channels
    df = df.loc[:, df.std(skipna=True) > 1e-8]
    # fill remaining gaps
    df = df.interpolate(method="linear", limit_direction="both").fillna(0.0)
    print(f"  Channels: {n0} -> {len(df.columns)}  (dropped {n0 - len(df.columns)})")
    return df


# ── SG filter: smooth + 1st and 2nd derivative ───────────────────────────────

def sg_features(df: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for col in tqdm(df.columns, desc="  SG filter"):
        s = df[col].values
        if len(s) >= SG_WINDOW:
            result[col]          = savgol_filter(s, SG_WINDOW, SG_POLYORDER, deriv=0)
            result[f"{col}_d1"]  = savgol_filter(s, SG_WINDOW, SG_POLYORDER, deriv=1)
            result[f"{col}_d2"]  = savgol_filter(s, SG_WINDOW, SG_POLYORDER, deriv=2)
        else:
            result[col]          = s
            result[f"{col}_d1"]  = np.gradient(s)
            result[f"{col}_d2"]  = np.gradient(np.gradient(s))
    return pd.DataFrame(result, index=df.index)


# ── rolling statistics ────────────────────────────────────────────────────────

def add_rolling_stats(sg_df: pd.DataFrame, base_cols: list) -> pd.DataFrame:
    extra = {}
    for col in base_cols:
        r = sg_df[col].rolling(ROLLING_WIN, min_periods=1)
        extra[f"{col}_rmean"] = r.mean().values
        extra[f"{col}_rstd"]  = r.std(ddof=0).fillna(0.0).values
    return pd.concat([sg_df, pd.DataFrame(extra, index=sg_df.index)], axis=1)


# ── per-timestamp label assignment ───────────────────────────────────────────

def build_label_series(idx: pd.DatetimeIndex, labels_merged: pd.DataFrame) -> pd.Series:
    arr = np.zeros(len(idx), dtype=np.int8)
    for _, row in labels_merged.iterrows():
        try:
            t0 = pd.Timestamp(row["StartTime"]).tz_localize(None)
            t1 = pd.Timestamp(row["EndTime"]).tz_localize(None)
        except Exception:
            continue
        mask = (idx >= t0) & (idx <= t1)
        # keep highest-priority class at each timestep
        arr[mask] = np.maximum(arr[mask], np.int8(row["class_id"]))
    return pd.Series(arr, index=idx, name="label")


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("\n[1/6] Loading metadata ...")
    labels_merged, channels_df = load_metadata()
    print(f"  Anomaly events: {len(labels_merged)}")

    print("\n[2/6] Loading telemetry ...")
    raw = load_target_channels(channels_df)
    print(f"  Raw shape: {raw.shape}")

    print("\n[3/6] Cleaning ...")
    raw = clean(raw)

    print("\n[4/6] Savitzky-Golay filter + derivatives ...")
    base_cols = list(raw.columns)
    feat = sg_features(raw)

    print("\n[5/6] Rolling statistics ...")
    feat = add_rolling_stats(feat, base_cols)
    print(f"  Feature matrix: {feat.shape}  "
          f"({len(base_cols)} channels x 5 features each)")

    print("\n[6/6] Labels, scaling, saving ...")
    feat["label"]      = build_label_series(feat.index, labels_merged)
    feat["class_name"] = feat["label"].map(CLASS_NAMES)

    # MinMax scale only numeric feature columns
    fcols = [c for c in feat.columns if c not in ("label", "class_name")]
    scaler = MinMaxScaler()
    feat[fcols] = scaler.fit_transform(feat[fcols])

    # ── summary ──────────────────────────────────────────────────────────────
    print("\nLabel distribution:")
    for cls_id, cnt in feat["label"].value_counts().sort_index().items():
        name = CLASS_NAMES[cls_id]
        bar  = "#" * int(50 * cnt / len(feat))
        print(f"  [{cls_id}] {name:<35s} {cnt:6d}  ({100*cnt/len(feat):5.1f}%)  {bar}")

    feat.index.name = "timestamp"
    feat.to_csv(OUTPUT_CSV)

    print(f"\nSaved -> {OUTPUT_CSV}")
    print(f"  Shape : {feat.shape}")
    print(f"  Features  : {len(fcols)}")
    print(f"  Timestamps: {len(feat)}\n")


if __name__ == "__main__":
    main()

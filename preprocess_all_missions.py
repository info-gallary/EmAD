"""
Multi-mission preprocessing pipeline.
Produces per-mission CSVs and a zero-padded combined dataset.

Outputs
-------
  data/mission1_preprocessed.csv   (275 features)
  data/mission2_preprocessed.csv   (~220 features)
  data/mission3_preprocessed.csv   (~120 features)
  data/all_missions_combined.csv   (all padded to MAX_FEATURES=275, + mission_id col)
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ---- mission configs --------------------------------------------------------
ESA_ROOT = r"d:\UbtVM-Def\Models\ESA-data"
OUT_DIR  = r"d:\UbtVM-Def\Models\data"

MISSIONS = {
    1: {"start": "2004-12-01", "end": "2004-12-15"},
    2: {"start": "2002-12-16", "end": "2002-12-31"},
    3: {"start": "2000-12-14", "end": "2000-12-28"},
}

RESAMPLE_FREQ = "60s"
SG_WINDOW     = 11
SG_POLYORDER  = 2
ROLLING_WIN   = 10
MISSING_THR   = 0.90

SUBSYSTEM_MAP = {
    "subsystem_1": "communication",
    "subsystem_5": "power",
    "subsystem_6": "thermal",
    "subsystem_3": "software",
}
CLASS_NAMES = {
    0: "Normal", 1: "Communication Anomaly", 2: "Power Anomaly",
    3: "Thermal Anomaly", 4: "Software Anomaly",
    5: "Rare Nominal Event", 6: "Communication Gap", 7: "Unknown Anomaly",
}


def base_dir(mission_id):
    return os.path.join(ESA_ROOT, f"ESA-Mission{mission_id}", f"ESA-Mission{mission_id}")


# ---- metadata ---------------------------------------------------------------

def load_metadata(base):
    types_df    = pd.read_csv(os.path.join(base, "anomaly_types.csv"))
    channels_df = pd.read_csv(os.path.join(base, "channels.csv"))
    labels_df   = pd.read_csv(os.path.join(base, "labels.csv"))

    merged = (labels_df
              .merge(types_df, on="ID")
              .merge(channels_df[["Channel", "Subsystem"]], on="Channel"))

    def assign_class(row):
        cat = row.get("Category", "")
        sub = row.get("Subsystem", "")
        if cat == "Rare Event":      return 5
        if cat == "Communication Gap": return 6
        if cat == "Anomaly":
            sem = SUBSYSTEM_MAP.get(sub, "unknown")
            return {"communication": 1, "power": 2, "thermal": 3, "software": 4}.get(sem, 7)
        return 7

    merged["class_id"] = merged.apply(assign_class, axis=1)
    return merged, channels_df


# ---- channel loading --------------------------------------------------------

def load_channels(base, channels_df, start, end):
    targets = channels_df[channels_df["Target"].str.strip().str.upper() == "YES"]["Channel"].tolist()
    frames  = []
    for ch in tqdm(targets, desc=f"  Loading channels", leave=False):
        path = os.path.join(base, "channels", ch, ch)
        if not os.path.exists(path):
            continue
        try:
            raw = pd.read_pickle(path)
            col = raw.iloc[:, 0]
            # handle categorical (object) channels via label encoding
            if col.dtype == object:
                col = col.astype("category").cat.codes.astype("float32")
            else:
                col = col.astype("float32")
            df = col.to_frame(name=ch)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index().loc[pd.Timestamp(start):pd.Timestamp(end)]
            if df.empty:
                continue
            df = df.resample(RESAMPLE_FREQ).mean()
            frames.append(df)
        except Exception as exc:
            pass
    if not frames:
        raise RuntimeError(f"No channels loaded for {base}")
    return pd.concat(frames, axis=1)


# ---- cleaning ---------------------------------------------------------------

def clean(df):
    n0 = len(df.columns)
    df = df.loc[:, df.isna().mean() < MISSING_THR]
    df = df.loc[:, df.std(skipna=True) > 1e-8]
    df = df.interpolate(method="linear", limit_direction="both").fillna(0.0)
    print(f"  Channels: {n0} -> {len(df.columns)}  (dropped {n0 - len(df.columns)})")
    return df


# ---- SG features ------------------------------------------------------------

def sg_features(df):
    result = {}
    for col in df.columns:
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


def add_rolling(sg_df, base_cols):
    extra = {}
    for col in base_cols:
        r = sg_df[col].rolling(ROLLING_WIN, min_periods=1)
        extra[f"{col}_rmean"] = r.mean().values
        extra[f"{col}_rstd"]  = r.std(ddof=0).fillna(0.0).values
    return pd.concat([sg_df, pd.DataFrame(extra, index=sg_df.index)], axis=1)


# ---- labels -----------------------------------------------------------------

def build_labels(idx, labels_merged):
    arr = np.zeros(len(idx), dtype=np.int8)
    for _, row in labels_merged.iterrows():
        try:
            t0 = pd.Timestamp(row["StartTime"]).tz_localize(None)
            t1 = pd.Timestamp(row["EndTime"]).tz_localize(None)
        except Exception:
            continue
        mask = (idx >= t0) & (idx <= t1)
        arr[mask] = np.maximum(arr[mask], np.int8(row["class_id"]))
    return pd.Series(arr, index=idx, name="label")


# ---- per-mission pipeline ---------------------------------------------------

def process_mission(mission_id):
    cfg   = MISSIONS[mission_id]
    base  = base_dir(mission_id)
    print(f"\n{'='*58}")
    print(f"  Mission {mission_id}  [{cfg['start']} -> {cfg['end']}]")
    print(f"{'='*58}")

    labels_merged, channels_df = load_metadata(base)
    print(f"  Anomaly events: {len(labels_merged)}")

    raw = load_channels(base, channels_df, cfg["start"], cfg["end"])
    print(f"  Raw shape: {raw.shape}")

    raw  = clean(raw)
    base_cols = list(raw.columns)
    feat = sg_features(raw)
    feat = add_rolling(feat, base_cols)

    feat["label"]      = build_labels(feat.index, labels_merged)
    feat["class_name"] = feat["label"].map(CLASS_NAMES)
    feat["mission_id"] = mission_id

    # label distribution
    print("  Label distribution:")
    for cls_id, cnt in feat["label"].value_counts().sort_index().items():
        print(f"    [{cls_id}] {CLASS_NAMES[cls_id]:<32s} {cnt:6d}  ({100*cnt/len(feat):5.1f}%)")

    # scale features
    fcols = [c for c in feat.columns if c not in ("label", "class_name", "mission_id")]
    scaler = MinMaxScaler()
    feat[fcols] = scaler.fit_transform(feat[fcols])

    out = os.path.join(OUT_DIR, f"mission{mission_id}_preprocessed.csv")
    feat.index.name = "timestamp"
    feat.to_csv(out)
    print(f"  Saved -> {out}  shape={feat.shape}")
    return feat, len(fcols)


# ---- combined dataset with zero-padding ------------------------------------

def combine_missions(dfs_with_ncols):
    max_feat = max(n for _, n in dfs_with_ncols)
    print(f"\nCombining missions (padding to {max_feat} features) ...")
    combined_frames = []
    for df, n in dfs_with_ncols:
        fcols = [c for c in df.columns if c not in ("label", "class_name", "mission_id")]
        meta  = df[["label", "class_name", "mission_id"]].reset_index(drop=True)
        feat  = df[fcols].reset_index(drop=True)
        # rename to generic feat_0..feat_N so concat aligns correctly
        feat.columns = [f"feat_{i}" for i in range(len(fcols))]
        # zero-pad up to max_feat
        if len(fcols) < max_feat:
            pad = pd.DataFrame(
                np.zeros((len(feat), max_feat - len(fcols)), dtype=np.float32),
                columns=[f"feat_{i}" for i in range(len(fcols), max_feat)]
            )
            feat = pd.concat([feat, pad], axis=1)
        combined_frames.append(pd.concat([feat, meta], axis=1))
    out_df = pd.concat(combined_frames, axis=0).reset_index(drop=True)
    out_df.index.name = "sample_id"
    return out_df


# ---- main -------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    for mid in [1, 2, 3]:
        df, n_feat = process_mission(mid)
        results.append((df, n_feat))

    combined = combine_missions(results)
    out = os.path.join(OUT_DIR, "all_missions_combined.csv")
    combined.to_csv(out)
    print(f"\nCombined dataset saved -> {out}")
    print(f"  Shape: {combined.shape}")
    print(f"  Missions: {combined['mission_id'].value_counts().to_dict()}")
    print(f"\nLabel distribution (combined):")
    for cls_id, cnt in combined["label"].value_counts().sort_index().items():
        print(f"  [{cls_id}] {CLASS_NAMES[cls_id]:<32s} {cnt:6d}  ({100*cnt/len(combined):5.1f}%)")


if __name__ == "__main__":
    main()

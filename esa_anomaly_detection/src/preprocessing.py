import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def clean_data(df, missing_threshold=0.9):
    """
    Remove invalid/empty channels and handle missing values.
    """
    print(f"Starting cleaning with {len(df.columns)} channels...")
    
    # 1. Remove channels with too many missing values FIRST
    miss_rates = df.isnull().mean()
    valid_cols = df.columns[miss_rates < missing_threshold]
    df = df[valid_cols]
    print(f"Channels after missing threshold ({missing_threshold}): {len(df.columns)}")
    
    if len(df.columns) == 0:
        return df

    # 2. Remove constant channels
    def is_constant(s):
        unique_vals = s.dropna().unique()
        return len(unique_vals) <= 1

    dynamic_cols = [col for col in df.columns if not is_constant(df[col])]
    df = df[dynamic_cols]
    print(f"Channels after removing constant ones: {len(df.columns)}")
    
    # 3. Fill remaining missing values with linear interpolation
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.fillna(0)
    
    return df

def apply_smoothing(df, window_length=11, polyorder=2):
    """
    Apply Savitzky–Golay smoothing filter.
    """
    print("Applying Savitzky–Golay smoothing...")
    smoothed_df = df.copy()
    for col in tqdm(df.columns):
        smoothed_df[col] = savgol_filter(df[col], window_length, polyorder)
    return smoothed_df

def create_sliding_windows(data, window_size=50, step=10):
    """
    Create sliding windows from the time series.
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def assign_multiclass_labels(df, labels_merged, window_size=50, step=10):
    """
    Assign multiclass anomaly labels (0-7) to each window.
    If multiple anomalies overlap a window, the one with the highest class priority is chosen.
    """
    # Create label map for the whole time series
    # Initialize with 0 (Normal)
    label_series = np.zeros(len(df), dtype=int)
    
    for _, row in labels_merged.iterrows():
        start = pd.to_datetime(row['StartTime']).tz_localize(None)
        end = pd.to_datetime(row['EndTime']).tz_localize(None)
        class_id = row['class_id']
        
        # Find indices within the interval
        mask = (df.index >= start) & (df.index <= end)
        # Update labels where they were 0 or have lower priority
        # (Assuming classes 1-4 are higher priority than 5-7)
        # For simplicity, we'll just take the first non-zero or max
        label_series[mask] = np.maximum(label_series[mask], class_id)
    
    # Label windows: take the mode or max class per window
    window_labels = []
    for i in range(0, len(df) - window_size + 1, step):
        window_pts = label_series[i:i + window_size]
        if np.any(window_pts > 0):
            # Take highest class ID as the window label
            window_labels.append(np.max(window_pts))
        else:
            window_labels.append(0)
    
    return np.array(window_labels)

def normalize_features(data):
    """
    Normalize features to [0, 1] range.
    """
    scaler = MinMaxScaler()
    if len(data.shape) == 3:
        n_windows, size, n_features = data.shape
        data_2d = data.reshape(-1, n_features)
        data_norm_2d = scaler.fit_transform(data_2d)
        return data_norm_2d.reshape(n_windows, size, n_features), scaler
    else:
        return scaler.fit_transform(data), scaler

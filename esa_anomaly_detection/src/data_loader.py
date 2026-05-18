import pandas as pd
import os
from tqdm import tqdm

# Mapping of subsystems to user categories (Inferred/Configurable)
SUBSYSTEM_MAP = {
    'subsystem_1': 'communication',
    'subsystem_5': 'power',
    'subsystem_6': 'thermal',
    'subsystem_3': 'software',
}

# Mapping of categories to class IDs
CLASS_MAP = {
    'normal': 0,
    'communication anomaly': 1,
    'power/electrical anomaly': 2,
    'thermal anomaly': 3,
    'software/reset/computer anomaly': 4,
    'rare nominal event': 5,
    'communication gap': 6,
    'unknown anomaly': 7
}

def load_mission_metadata(mission_dir):
    """
    Loads and merges metadata to create an anomaly-to-class mapping.
    """
    types_path = os.path.join(mission_dir, "anomaly_types.csv")
    channels_path = os.path.join(mission_dir, "channels.csv")
    labels_path = os.path.join(mission_dir, "labels.csv")
    
    if not all(os.path.exists(p) for p in [types_path, channels_path, labels_path]):
        return None, None
    
    types_df = pd.read_csv(types_path)
    channels_df = pd.read_csv(channels_path)
    labels_df = pd.read_csv(labels_path)
    
    # Merge labels with anomaly types
    labels_merged = labels_df.merge(types_df, on='ID')
    # Merge with channels to get subsystem
    labels_merged = labels_merged.merge(channels_df[['Channel', 'Subsystem']], on='Channel')
    
    def map_to_class(row):
        cat = row['Category']
        subsys = row['Subsystem']
        
        if cat == 'Rare Event':
            return CLASS_MAP['rare nominal event']
        elif cat == 'Communication Gap':
            return CLASS_MAP['communication gap']
        elif cat == 'Anomaly':
            subsys_type = SUBSYSTEM_MAP.get(subsys, 'unknown')
            if subsys_type == 'communication':
                return CLASS_MAP['communication anomaly']
            elif subsys_type == 'power':
                return CLASS_MAP['power/electrical anomaly']
            elif subsys_type == 'thermal':
                return CLASS_MAP['thermal anomaly']
            elif subsys_type == 'software':
                return CLASS_MAP['software/reset/computer anomaly']
            else:
                return CLASS_MAP['unknown anomaly']
        else:
            return CLASS_MAP['unknown anomaly']

    labels_merged['class_id'] = labels_merged.apply(map_to_class, axis=1)
    
    return labels_merged, channels_df

def load_mission_data(mission_dir, channel_ids=None, start_time=None, end_time=None, resample_freq='60s'):
    """
    Loads telemetry channel data for a given mission with time-based slicing and resampling.
    """
    channels_path = os.path.join(mission_dir, "channels")
    if not channel_ids:
        channel_ids = [d for d in os.listdir(channels_path) if os.path.isdir(os.path.join(channels_path, d))]
    
    dataframes = []
    print(f"Loading and resampling {len(channel_ids)} channels...")
    
    for ch_id in tqdm(channel_ids):
        ch_file = os.path.join(channels_path, ch_id, ch_id)
        if os.path.exists(ch_file):
            try:
                df = pd.read_pickle(ch_file).astype('float32')
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                if start_time and end_time:
                    df = df.loc[start_time : end_time]
                
                if df.empty:
                    continue
                
                df = df.resample(resample_freq).mean()
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {ch_id}: {e}")
    
    if not dataframes:
        return None
    
    full_df = pd.concat(dataframes, axis=1)
    return full_df

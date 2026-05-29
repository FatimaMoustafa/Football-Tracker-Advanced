import pandas as pd
import numpy as np
import os

def compute_raw_speed(df, fps):
    df['dx'] = df.groupby('player_id')['x_m'].diff()
    df['dy'] = df.groupby('player_id')['y_m'].diff()
    df["delta_m"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["speed_mps"] = df["delta_m"] * fps
    df['speed_mps'] = df['speed_mps'].fillna(0.0)
    return df

def add_smoothing(df):
    # smooth the speed by taking the median of each 5 frames for each player in each frame
    spikes = df[df['speed_mps'] > 13.0]
    df["speed_smoothed"] = df.groupby('player_id')['speed_mps'].transform(lambda x: x.rolling(5, center=True).median())
    df["speed_smoothed"] = np.minimum(df["speed_smoothed"],13.0)
    for _,row in spikes.iterrows():
        print(f"[spike] player_id = {row['player_id']} frame={row['frame']} speed={row['speed_mps']:.2f} m/s")

    df["speed_smoothed"] = df["speed_smoothed"].fillna(df['speed_mps'])
    return df

def export_df(df):
    output_df = df[['player_id','team_id','frame','time','x_m','y_m','speed_smoothed']]
    os.makedirs('outputs', exist_ok=True)
    output_df.to_csv('outputs/tracking_output.csv', index=False)
    return output_df
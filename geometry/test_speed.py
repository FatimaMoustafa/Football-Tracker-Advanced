import numpy as np
import sys
sys.path.append('d:/Level 4/final_project/geometry')

from homography import build_homography, load_or_pick_points, pixel_to_meters
from speed import compute_raw_speed, add_smoothing, export_df

# ── Step 1: Load H matrix ─────────────────────────────────────────────────
H = np.load('calibrations/H_matrix.npy')

# ── Step 2: Load and enrich tracking data ─────────────────────────────────
from tracker_output import enrich_tracks   # your Day 4 file
df = enrich_tracks(json_path='data.json', H=H)

# ── Step 3: Compute raw speed ─────────────────────────────────────────────
df = compute_raw_speed(df, fps=25)
print(df[['player_id', 'frame', 'x_m', 'y_m', 'speed_mps']].head(10))

# ── Step 4: Add smoothing ─────────────────────────────────────────────────
df = add_smoothing(df)
print(f"\nMax speed after smoothing: {df['speed_smoothed'].max():.2f} m/s")
print(df[['player_id', 'frame', 'speed_mps', 'speed_smoothed']].head(10))

# ── Step 5: Export ────────────────────────────────────────────────────────
export_df(df)
print("\nCSV exported successfully ✅")
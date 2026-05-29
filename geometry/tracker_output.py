import json
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from homography import pixel_to_meters


# FLATTEN JSON → DataFrame# 

def load_tracks(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    for player in data["players"]:
        pid        = player["id"]
        team_id    = player["team_id"]
        jersey     = player["jersey_number"]

        for entry in player["tracking"]:
            rows.append({
                "player_id":     pid,
                "team_id":       team_id,
                "jersey_number": jersey,
                "frame":         entry["frame"],
                "time":          entry["time"],
                "x_fake_m":      entry["x"],
                "y_fake_m":      entry["y"],
                "speed_mps":     entry["speed_mps"],
                "vx":            entry["velocity"]["vx"],
                "vy":            entry["velocity"]["vy"],
                "confidence":    entry["confidence"],
                "interpolated":  entry["interpolated"],
                "direction_deg": entry["direction_deg"],
            })

    df = pd.DataFrame(rows)
    print(f"[load_tracks] Loaded {len(df)} rows — "
          f"{df['player_id'].nunique()} players, "
          f"{df['frame'].nunique()} frames")
    return df


# 2. SIMULATE PIXEL COORDINATES  (fake-data only — real pipeline skips this)
def meters_to_pixels(x_m: float, y_m: float, H: np.ndarray):
    H_inv = np.linalg.inv(H)
    pt    = np.array([[[x_m, y_m]]], dtype=np.float32)
    px    = cv2.perspectiveTransform(pt, H_inv)
    return float(px[0, 0, 0]), float(px[0, 0, 1])


def add_simulated_pixels(df: pd.DataFrame, H: np.ndarray):
    px_coords = df.apply(
        lambda row: meters_to_pixels(row["x_fake_m"], row["y_fake_m"], H),
        axis=1
    )
    df = df.copy()
    df["foot_x_px"] = px_coords.apply(lambda t: float(t[0]))
    df["foot_y_px"] = px_coords.apply(lambda t: float(t[1]))

    print(f"[add_simulated_pixels] Pixel range → "
          f"x: [{df['foot_x_px'].min():.1f}, {df['foot_x_px'].max():.1f}]  "
          f"y: [{df['foot_y_px'].min():.1f}, {df['foot_y_px'].max():.1f}]")
    return df


# 3. APPLY HOMOGRAPHY → x_m, y_m
def add_metric_coords(df: pd.DataFrame, H: np.ndarray):
    metric = df.apply(
        lambda row: pixel_to_meters(row["foot_x_px"], row["foot_y_px"], H),
        axis=1
    )
    df = df.copy()
    df["x_m"] = metric.apply(lambda t: round(t[0], 3))
    df["y_m"] = metric.apply(lambda t: round(t[1], 3))

    print(f"[add_metric_coords] Metric range → "
          f"x_m: [{df['x_m'].min():.2f}, {df['x_m'].max():.2f}]  "
          f"y_m: [{df['y_m'].min():.2f}, {df['y_m'].max():.2f}]")
    return df




# 4. ROUND-TRIP VALIDATION
def validate_round_trip(df: pd.DataFrame, tolerance_m: float = 0.01):
    df = df.copy()
    df["error_m"] = np.sqrt(
        (df["x_m"] - df["x_fake_m"]) ** 2 +
        (df["y_m"] - df["y_fake_m"]) ** 2
    )

    max_err  = df["error_m"].max()
    mean_err = df["error_m"].mean()

    print(f"\n[validate_round_trip]")
    print(f"  Mean error : {mean_err*100:.4f} cm")
    print(f"  Max error  : {max_err*100:.4f} cm")

    if max_err < tolerance_m:
        print(f"  Round-trip valid ✅ — all errors < {tolerance_m*100:.0f} cm")
    else:
        bad = df[df["error_m"] >= tolerance_m]
        print(f"  ⚠️  {len(bad)} rows exceeded tolerance — check H matrix")
        print(bad[["player_id", "frame", "x_fake_m", "y_fake_m",
                    "x_m", "y_m", "error_m"]].head(5).to_string(index=False))


# 5. SANITY PLOT
def sanity_plot(df: pd.DataFrame, save_path: str = "outputs/sanity_plot.png"):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#3a7d44")          # pitch green

    # Draw pitch outline
    pitch = patches.Rectangle((0, 0), 105, 68,
                               linewidth=2, edgecolor="white", facecolor="none")
    ax.add_patch(pitch)

    # halfway line
    ax.plot([52.5, 52.5], [0, 68], color="white", linewidth=1.5)

    # centre circle (radius 9.15m)
    centre = plt.Circle((52.5, 34), 9.15,
                         color="white", fill=False, linewidth=1.5)
    ax.add_patch(centre)

    # left penalty box
    ax.add_patch(patches.Rectangle((0, 13.84), 16.5, 40.32,
                 linewidth=1.5, edgecolor="white", facecolor="none"))
    # right penalty box
    ax.add_patch(patches.Rectangle((88.5, 13.84), 16.5, 40.32,
                 linewidth=1.5, edgecolor="white", facecolor="none"))

    # Plot players
    colours = {1: "#e63946", 2: "#1d6fa4"}   # Team 1 = red, Team 2 = blue
    labels  = {1: "Team A", 2: "Team B"}

    for team_id, group in df.groupby("team_id"):
        ax.scatter(
            group["x_m"], group["y_m"],
            c=colours[team_id], label=labels[team_id],
            alpha=0.55, s=30, edgecolors="white", linewidths=0.4
        )

    # Highlight interpolated points
    interp = df[df["interpolated"] == True]
    ax.scatter(interp["x_m"], interp["y_m"],
               facecolors="none", edgecolors="yellow",
               s=80, linewidths=1.5, label="Interpolated", zorder=5)

    ax.set_xlim(-3, 108)
    ax.set_ylim(-3, 71)
    ax.set_xlabel("x (meters)", color="white")
    ax.set_ylabel("y (meters)", color="white")
    ax.set_title("Sanity Plot — Tracker Output in Metric Space", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("white")
    fig.patch.set_facecolor("#1e1e2e")
    ax.legend(facecolor="#1e1e2e", labelcolor="white", loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[sanity_plot] Saved to {save_path}")



# 6. MAIN ENTRY POINT
def enrich_tracks(json_path: str, H: np.ndarray):
    df = load_tracks(json_path)
    df = add_simulated_pixels(df, H)
    df = add_metric_coords(df, H)
    validate_round_trip(df)
    sanity_plot(df, save_path="outputs/sanity_plot.png")
    return df

if __name__ == "__main__":
    H = np.load('calibrations/H_matrix.npy')

    df = enrich_tracks(
        json_path="data.json",
        H=H
    )
    print(df.columns)

    print(df.head())
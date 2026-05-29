# MatchIQ — Geometry & Physics Module

**Owner:** Geometry / Physics Engineer  
**Module path:** `geometry/`  
**Status:** ✅ Complete — Days 1–7

---

## Overview

This module transforms raw pixel coordinates from the tracker into real-world metric coordinates (meters) on a standard FIFA 105×68m pitch. It also computes per-player speed in m/s with temporal smoothing and physical constraints.

---

## File Structure

```
geometry/
├── homography.py       # Core transform: pixel → meter
├── speed.py            # Speed engine: raw speed, smoothing, export
├── plan_b.py           # Fallback transform: HSV field mask scaling
├── minimap.py          # Bird's-eye minimap overlay for video output
├── config.py           # Plan A vs Plan B flag per clip
└── calibrations/
    ├── wide_01.json    # Saved calibration pixel + world points
    └── H_matrix.npy   # Saved 3×3 homography matrix
```

---

## How to Calibrate (Plan A — Homography)

**Step 1 — Choose a frame with 4 visible pitch landmarks**

Good landmarks (well spread, not collinear):
- Left penalty spot → `(11.0, 34.0)`
- Top-left of left penalty box → `(0.0, 13.84)`
- Bottom-right of left penalty box → `(16.5, 54.16)`
- Center spot → `(52.5, 34.0)`

> ⚠️ Never pick 3 points that share the same X or Y value — collinear points cause the homography to collapse.

**Step 2 — Update `DEFAULT_WORLD_POINTS` in `homography.py`**

```python
DEFAULT_WORLD_POINTS = np.array([
    [11.0,  34.0],    # left penalty spot
    [ 0.0,  13.84],   # top-left of left penalty box
    [16.5,  54.16],   # bottom-right of left penalty box
    [52.5,  34.0],    # center spot
], dtype=np.float32)
```

**Step 3 — Delete old calibration and re-run**

```powershell
Remove-Item geometry/calibrations/wide_01.json
python geometry/homography.py
```

A window opens — click the 4 landmarks in the exact order matching `DEFAULT_WORLD_POINTS`. The calibration is saved automatically.

**Step 4 — Verify**

The script runs `validate_transform()` automatically. All errors must be < 0.5m.

---

## How to Run the Full Pipeline

```python
import numpy as np
from homography import load_or_pick_points, build_homography, pixel_to_meters, validate_transform
from speed import compute_raw_speed, add_smoothing, export_df

# 1. Load frame and calibrate
frame = cv2.imread('your_clip_frame.png')
frame = cv2.resize(frame, (1280, 720))
pixel_pts, world_pts = load_or_pick_points(frame, save_path='calibrations/wide_01.json')

# 2. Build homography
H = build_homography(pixel_pts, world_pts)

# 3. Convert any pixel to meters
x_m, y_m = pixel_to_meters(x_px, y_px, H)

# 4. Process tracker DataFrame
df = compute_raw_speed(df, fps=25)
df = add_smoothing(df)
export_df(df)   # saves to outputs/tracking_output.csv
```

---

## Coordinate System

| Parameter | Value |
|---|---|
| Origin (0, 0) | Top-left corner of pitch |
| X-axis | Along pitch length → 0 to 105m |
| Y-axis | Along pitch width → 0 to 68m |
| Center spot | (52.5, 34.0) |
| Left penalty spot | (11.0, 34.0) |
| Right penalty spot | (94.0, 34.0) |

---

## Speed Engine

| Parameter | Value |
|---|---|
| Method | Euclidean distance between consecutive frames |
| Formula | `delta_m * fps` |
| Smoothing | 5-frame rolling median per player |
| Max speed cap | 13.0 m/s (~47 km/h) |
| Spike logging | Any raw speed > 13 m/s is printed with player_id and frame |

---

## Output Format

`outputs/tracking_output.csv` — one row per player per frame:

| Column | Type | Description |
|---|---|---|
| `player_id` | int | Unique player identifier |
| `team_id` | int | Team (1 or 2) |
| `frame` | int | Frame number |
| `time` | float | Timestamp in seconds |
| `x_m` | float | X position in meters |
| `y_m` | float | Y position in meters |
| `speed_smoothed` | float | Smoothed speed in m/s |

---

## Plan A vs Plan B

| | Plan A (Homography) | Plan B (Field Mask) |
|---|---|---|
| **Method** | `cv2.getPerspectiveTransform` on 4 landmarks | HSV green mask → bounding box → linear scale |
| **Accuracy** | High (~0.0m error on calibration points) | Low (~±5m depending on camera angle) |
| **When to use** | Clear pitch landmarks visible | Narrow angle, landmarks off-frame |
| **Function** | `pixel_to_meters(x, y, H)` | `plan_b_pixel_to_meters(frame, x, y)` |

To set which plan a clip uses, edit `config.py`:

```python
CLIP_CONFIG = {
    "clip_01": "plan_a",   # homography — landmarks visible
    "clip_02": "plan_b",   # field mask fallback — bad angle
}
```

---

## Dependencies

```
opencv-python
numpy
pandas
scipy
matplotlib
```

Install with:
```bash
pip install opencv-python numpy pandas scipy matplotlib
```

---

## Validation Checklist

Before handing off to teammates:

- [ ] `validate_transform()` passes — all errors < 0.5m
- [ ] `speed_smoothed` max < 13.0 m/s
- [ ] `x_m` range: 0–105, `y_m` range: 0–68
- [ ] `tracking_output.csv` has correct columns
- [ ] Sanity plot shows player cloud inside pitch boundaries
import cv2
import numpy as np
import pandas as pd


# Minimap configuration
MINIMAP_WIDTH  = 210          # pixels — 2:1 scale of 105m
MINIMAP_HEIGHT = 136          # pixels — 2:1 scale of 68m
MINIMAP_MARGIN = 10           # pixels — padding from frame corner
MINIMAP_ALPHA  = 0.85         # opacity of minimap overlay

TEAM_COLORS = {
    1: (60,  60,  220),       # Team 1 — red   (BGR)
    2: (220, 100,  30),       # Team 2 — blue  (BGR)
}
DOT_RADIUS    = 4
DOT_THICKNESS = -1            # filled circle


def _build_pitch_canvas():
    canvas = np.full((MINIMAP_HEIGHT, MINIMAP_WIDTH, 3), (40, 100, 40), dtype=np.uint8)

    w, h = MINIMAP_WIDTH, MINIMAP_HEIGHT

    # Pitch outline
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)

    # Halfway line
    cv2.line(canvas, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

    # Centre circle (radius 9.15m → scaled)
    r_x = int(9.15 / 105.0 * w)
    r_y = int(9.15 / 68.0  * h)
    cv2.ellipse(canvas, (w // 2, h // 2), (r_x, r_y), 0, 0, 360, (255, 255, 255), 1)

    # Centre spot
    cv2.circle(canvas, (w // 2, h // 2), 2, (255, 255, 255), -1)

    # Left penalty box (0–16.5m × 13.84–54.16m)
    lp_x2 = int(16.5  / 105.0 * w)
    lp_y1 = int(13.84 / 68.0  * h)
    lp_y2 = int(54.16 / 68.0  * h)
    cv2.rectangle(canvas, (0, lp_y1), (lp_x2, lp_y2), (255, 255, 255), 1)

    # Right penalty box (88.5–105m × 13.84–54.16m)
    rp_x1 = int(88.5  / 105.0 * w)
    rp_y1 = int(13.84 / 68.0  * h)
    rp_y2 = int(54.16 / 68.0  * h)
    cv2.rectangle(canvas, (rp_x1, rp_y1), (w - 1, rp_y2), (255, 255, 255), 1)

    return canvas


def _meters_to_minimap(x_m: float, y_m: float):
    px = int((x_m / 105.0) * MINIMAP_WIDTH)
    py = int((y_m / 68.0)  * MINIMAP_HEIGHT)

    # clamp to canvas bounds
    px = max(0, min(px, MINIMAP_WIDTH  - 1))
    py = max(0, min(py, MINIMAP_HEIGHT - 1))

    return px, py


def draw_minimap(frame: np.ndarray, df: pd.DataFrame, frame_number: int):
    output = frame.copy()

    # Build pitch canvas
    minimap = _build_pitch_canvas()

    # Filter to current frame
    current = df[df['frame'] == frame_number]

    # Draw each player as a dot
    for _, row in current.iterrows():
        px, py   = _meters_to_minimap(row['x_m'], row['y_m'])
        team_id  = int(row['team_id'])
        color    = TEAM_COLORS.get(team_id, (200, 200, 200))

        cv2.circle(minimap, (px, py), DOT_RADIUS, color, DOT_THICKNESS)

        # white border around dot for visibility
        cv2.circle(minimap, (px, py), DOT_RADIUS, (255, 255, 255), 1)

    # Overlay minimap onto frame (bottom-left corner)
    frame_h, frame_w = output.shape[:2]

    y1 = frame_h - MINIMAP_HEIGHT - MINIMAP_MARGIN
    y2 = frame_h - MINIMAP_MARGIN
    x1 = MINIMAP_MARGIN
    x2 = MINIMAP_MARGIN + MINIMAP_WIDTH

    # alpha blend minimap onto frame
    roi = output[y1:y2, x1:x2]
    blended = cv2.addWeighted(minimap, MINIMAP_ALPHA, roi, 1 - MINIMAP_ALPHA, 0)
    output[y1:y2, x1:x2] = blended

    return output
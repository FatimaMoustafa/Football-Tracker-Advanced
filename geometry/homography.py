import cv2
import numpy as np
import json
import os
import math


DEFAULT_WORLD_POINTS = np.array([
    [11.0,  34.0],    # left penalty spot
    [ 0.0,  13.84],   # top-left corner of left penalty box
    [16.5,  54.16],   # bottom-right corner of left penalty box
    [52.5,  34.0],    # center spot
], dtype=np.float32)

def pick_points(frame, save_path, world_points=DEFAULT_WORLD_POINTS):
    """
    Opens a CV2 window, lets user click 4 pitch landmark points,
    draws numbered circles as visual feedback, saves pixel + world
    points to JSON, and returns pixel points as float32 numpy array.

    Args:
        frame:        np.ndarray (H, W, 3) — video frame to display
        save_path:    str — path to save JSON e.g. 'calibrations/wide_01.json'
        world_points: np.ndarray (4, 2) float32 — real-world meter coords
                      matching the order you will click the pixel points

    Returns:
        np.ndarray shape (4, 2) dtype float32 — pixel points
    """
    points = []          # collected pixel points — shared with callback
    original = frame.copy()   # keep a clean copy for redrawing each loop

    def click_handler(event, x, y, flags, param):
        """Registered once with setMouseCallback, fired by OpenCV on clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                print(f"  [pick_points] Point {len(points)}/4 → pixel ({x}, {y})"
                      f"  ↔  world {world_points[len(points)-1].tolist()} m")

    cv2.imshow('Pick 4 Points', frame)
    cv2.setMouseCallback('Pick 4 Points', click_handler)

    while len(points) < 4:
        cv2.waitKey(1)

        # ── Redraw frame with all points collected so far ──────────────────
        display = original.copy()          # fresh copy every iteration

        for i, (px, py) in enumerate(points):
            # draw a filled circle at the clicked location
            cv2.circle(display, (px, py), radius=8,
                       color=(0, 255, 0), thickness=-1)

            # draw the point index number just above the circle
            cv2.putText(display, str(i + 1), (px + 10, py - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9, color=(0, 255, 0), thickness=2)

            # draw the world coordinate as a small label below the number
            world_label = f"{world_points[i].tolist()} m"
            cv2.putText(display, world_label, (px + 10, py + 15),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 255, 0), thickness=1)

        # show how many points remain
        remaining = 4 - len(points)
        cv2.putText(display, f"Click {remaining} more point(s)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), thickness=2)

        cv2.imshow('Pick 4 Points', display)

    cv2.destroyAllWindows()

    # ── Build and save JSON ────────────────────────────────────────────────
    pixel_pts = np.array(points, dtype=np.float32)

    payload = {
        "save_path":       save_path,
        "pixel_points":    pixel_pts.tolist(),       # [[px,py], ...]
        "world_points_m":  world_points.tolist(),    # [[Xm,Ym], ...]
        "point_labels": [
            "top_left", "top_right", "bottom_right", "bottom_left"
        ]
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"[pick_points] Saved to {save_path}")
    return pixel_pts


def load_or_pick_points(frame, save_path, world_points=DEFAULT_WORLD_POINTS):
    """
    Loads pixel + world points from JSON if the file exists.
    If not, calls pick_points() so the user can click them.

    This is the function your pipeline should always call — never
    call pick_points() directly from pipeline code.

    Args:
        frame:        np.ndarray — video frame (only used if clicking needed)
        save_path:    str — JSON path to load from or save to
        world_points: np.ndarray (4, 2) — only used if clicking needed

    Returns:
        pixel_pts:   np.ndarray (4, 2) float32
        world_pts:   np.ndarray (4, 2) float32
    """
    if os.path.exists(save_path):
        # ── Load from disk — no window opened ─────────────────────────────
        with open(save_path, 'r') as f:
            payload = json.load(f)

        # JSON loads as float64 by default → must cast to float32
        pixel_pts = np.array(payload["pixel_points"],   dtype=np.float32)
        world_pts = np.array(payload["world_points_m"], dtype=np.float32)

        print(f"[load_or_pick_points] Loaded from {save_path} — skipping click UI")
        return pixel_pts, world_pts

    else:
        # ── No JSON found — open click UI ─────────────────────────────────
        print(f"[load_or_pick_points] No file at {save_path} — opening click UI")
        pixel_pts = pick_points(frame, save_path, world_points)
        world_pts = world_points
        return pixel_pts, world_pts

def build_homography(px_points, world_points):
    # reshape the points to be consistent with how getPerspectiveTransform handles the input
    src = px_points.reshape(4,1,2).astype(np.float32)
    dst = world_points.reshape(4,1,2).astype(np.float32)

    save_path='calibrations/H_matrix.npy'
    # Call the cv2 getPerspectiveTransform function to get the Homography matrix
    H = cv2.getPerspectiveTransform(src,dst)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, H)
    return H

def pixel_to_meters(x_px, y_px, H):
    points = np.array([x_px,y_px]).reshape(1,1,2).astype(np.float32)
    mapped = cv2.perspectiveTransform(points,H)
    X_m, y_m = mapped[0,0]
    return X_m, y_m

def validate_transform(H, test_pts):
    errors = []
    for (x_px, y_px, x_m, y_m) in test_pts:
        x_pred, y_pred = pixel_to_meters(x_px, y_px, H)
        error = math.sqrt((x_pred - x_m)**2 + (y_pred - y_m)**2)
        errors.append(error)
        print(f"the error for point coordinates in pixels {x_px} , {y_px} is {error}" \
        f" since it has real world coordinates {x_m} , {y_m} and the prdicted coordinates are"\
        f"{x_pred}, {y_pred}")
    for error in errors:
        if error >= 0.5:
            print("Error in tranforming !")
            break
    else:
        print("Transform valid ✅ — all errors < 0.5m")





if __name__ == '__main__':
    frame = cv2.imread('4.png')
    if frame is None:
        raise FileNotFoundError(
            "geometry/homography.py demo requires '4.png' in the current working directory "
            "or a valid image path."
        )

    frame = cv2.resize(frame, (1280, 720))
    pixel_pts, world_pts = load_or_pick_points(
        frame,
        save_path='calibrations/wide_01.json'
    )

    print(pixel_pts)        # shape (4,2), float32
    print(pixel_pts.dtype)  # must print float32
    print(world_pts)        # the 4 real-world meter coordinates
    H = build_homography(pixel_pts, world_pts)
    x_m, y_m = pixel_to_meters(pixel_pts[0][0], pixel_pts[0][1], H)
    print(f"Pixel {pixel_pts[0]} → World ({x_m:.2f}, {y_m:.2f}) m")

    test_pts = [
        (pixel_pts[0][0], pixel_pts[0][1], world_pts[0][0], world_pts[0][1]),
        (pixel_pts[1][0], pixel_pts[1][1], world_pts[1][0], world_pts[1][1]),
        (pixel_pts[2][0], pixel_pts[2][1], world_pts[2][0], world_pts[2][1]),
        (pixel_pts[3][0], pixel_pts[3][1], world_pts[3][0], world_pts[3][1]),
    ]
    validate_transform(H, test_pts)
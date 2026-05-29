# MatchIQ — Geometry Transform Config

# This file documents which transform method is used per clip.
# Plan A — Homography (cv2.getPerspectiveTransform)
#    Use when: pitch landmarks (penalty box corners, center spot) are visible
#    Accuracy: ~0.0m error on calibration points
#    Function: pixel_to_meters(x_px, y_px, H) in homography.py
#
# Plan B — Field Mask Fallback
#   Use when: camera angle too narrow, landmarks off-frame
#   Accuracy: ~±5m depending on how much pitch is visible
#   Function: plan_b_pixel_to_meters(frame, x_px, y_px) in plan_b.py

CLIP_CONFIG = {
    # Calibrated clips (Plan A)
    "clip_01": "plan_a",   # wide angle — all 4 landmarks visible
    "clip_02": "plan_a",   # wide angle — penalty box corners clear

    # Fallback clips (Plan B)
    "clip_03": "plan_b",   # tight angle — landmarks off-frame
    "clip_04": "plan_b",   # close-up — only partial pitch visible
}


def get_transform_method(clip_name: str):
    method = CLIP_CONFIG.get(clip_name, "plan_a")

    if method not in ("plan_a", "plan_b"):
        raise ValueError(f"[config] Unknown method '{method}' for clip '{clip_name}'. "
                         f"Must be 'plan_a' or 'plan_b'.")

    print(f"[config] {clip_name} → using {method.upper()}")
    return method


def get_transform_function(clip_name: str, H=None, frame=None):
    method = get_transform_method(clip_name)

    if method == "plan_a":
        if H is None:
            raise ValueError("[config] Plan A requires H matrix — pass H=your_matrix")
        from homography import pixel_to_meters
        return lambda x_px, y_px: pixel_to_meters(x_px, y_px, H)

    else:  # plan_b
        if frame is None:
            raise ValueError("[config] Plan B requires frame — pass frame=your_frame")
        from plan_b import plan_b_pixel_to_meters
        return lambda x_px, y_px: plan_b_pixel_to_meters(frame, x_px, y_px)
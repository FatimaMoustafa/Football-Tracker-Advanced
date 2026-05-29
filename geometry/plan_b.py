import cv2
import numpy as np

def plan_b_pixel_to_meters(frame, x_px, y_px):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([35, 40, 40])
    upper_bound = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    x_min, y_min, width, height = cv2.boundingRect(largest)
    # linear scaling
    x_m = (x_px - x_min) / width  * 105.0
    y_m = (y_px - y_min) / height * 68.0
    return x_m , y_m
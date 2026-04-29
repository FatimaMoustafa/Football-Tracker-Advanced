import numpy as np
import cv2

class TeamAssigner:
    def __init__(self):
        self.team_colors = {
            1: (255, 255, 255),  # white team → white circles
            2: (0, 255, 0)       # green team → green circles
        }
        self.player_team_dict = {}

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([0, 0, 0])
        
        # Use middle vertical strip, top 40% to get jersey
        h, w = crop.shape[:2]
        jersey = crop[int(h*0.1):int(h*0.4), int(w*0.25):int(w*0.75)]
        if jersey.size == 0:
            return np.array([0, 0, 0])
        
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        avg_color = np.mean(jersey.reshape(-1, 3), axis=0)  # BGR
        return avg_color

    def assign_team_color(self, frame, player_detections):
        # Colors already fixed — nothing to compute
        pass

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        color = self.get_player_color(frame, player_bbox)
        b, g, r = color

        # White team: high brightness across all channels
        # Green team: green channel dominant
        brightness = np.mean(color)
        greenness = g - max(r, b)  # how much greener than other channels

        if brightness > 150:
            # Bright → white team
            team_id = 1
        elif greenness > 20:
            # Green dominant → green team
            team_id = 2
        else:
            # Dark/ambiguous → assign by which team color it's closer to
            white = np.array([255, 255, 255])
            green = np.array([0, 255, 0])
            dist_white = np.linalg.norm(color - white)
            dist_green = np.linalg.norm(color - green)
            team_id = 1 if dist_white < dist_green else 2

        self.player_team_dict[player_id] = team_id
        return team_id
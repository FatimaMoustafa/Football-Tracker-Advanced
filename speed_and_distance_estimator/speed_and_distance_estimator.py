import cv2
import sys 
sys.path.append('../')
from utils import measure_distance ,get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window=5
        self.frame_rate=24
    
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        last_positions = {}
        last_valid_speed = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue

            total_distance[object] = {}
            last_positions[object] = {}
            last_valid_speed[object] = {}

            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_info in frame_tracks.items():
                    position = track_info.get('position_transformed')
                    if position is None:
                        position = track_info.get('position_adjusted')
                    if position is None:
                        position = track_info.get('position')

                    if position is None:
                        continue

                    if track_id in last_positions[object] and last_positions[object][track_id] is not None:
                        prev_position = last_positions[object][track_id]
                        distance_covered = measure_distance(prev_position, position)
                        time_elapsed = 1.0 / self.frame_rate
                        speed_mps = distance_covered / max(time_elapsed, 1e-6)
                        speed_kmh = speed_mps * 3.6

                        # Ignore clearly invalid jumps due to detection noise or ID switches
                        if speed_kmh > 45 or distance_covered > 300:
                            speed_kmh = last_valid_speed[object].get(track_id, 0.0)
                            distance_covered = 0.0
                        else:
                            total_distance[object].setdefault(track_id, 0.0)
                            total_distance[object][track_id] += distance_covered
                            last_valid_speed[object][track_id] = speed_kmh

                        track_info['speed'] = speed_kmh
                        track_info['distance'] = total_distance[object].get(track_id, 0.0)
                    else:
                        total_distance[object].setdefault(track_id, 0.0)
                        track_info['speed'] = 0.0
                        track_info['distance'] = total_distance[object].get(track_id, 0.0)

                    last_positions[object][track_id] = position
    
    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed',None)
                       distance = track_info.get('distance',None)
                       if speed is None or distance is None:
                           continue
                       
                       bbox = track_info['bbox']
                       position = get_foot_position(bbox)
                       position = list(position)
                       position[1]+=40

                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} km/h",position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0],position[1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            output_frames.append(frame)
        
        return output_frames
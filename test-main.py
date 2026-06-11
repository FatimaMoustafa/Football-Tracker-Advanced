from pathlib import Path

from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from geometry.homography import load_or_pick_points, build_homography, pixel_to_meters
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from data.json_extractor import export_match_data
from visualizations.heatmap import heatmap_by_team, heatmap_for_team
from visualizations.pass_networks import pass_network
from visualizations.json_extractor import (
    assign_ball_touches,
    determine_ball_control,
    detect_passes,
    build_players,
)


def main():
    # Read Video
    video_path = 'input_videos/08fd33_4.mp4'
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    video_frames = read_video(video_path)
    if len(video_frames) == 0:
        raise ValueError(f"Input video contains no frames: {video_path}")

   
    max_frames = 700  # Change this value to process more/fewer frames
    video_frames = video_frames[:max_frames]

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    # Load or generate tracks, with validation for frame count mismatch
    stub_path = 'stubs/track_stubs.pkl'
    use_stub = True
    
    # Verify stub data matches current frame count
    if Path(stub_path).exists():
        import pickle
        try:
            with open(stub_path, 'rb') as f:
                cached_tracks = pickle.load(f)
            # Check if cached tracks have matching frame count
            if len(cached_tracks.get("players", [])) != len(video_frames):
                print(f"Frame count mismatch: cached {len(cached_tracks['players'])} frames vs current {len(video_frames)} frames")
                print("Regenerating tracks...")
                use_stub = False
        except Exception as e:
            print(f"Error reading stub file: {e}. Regenerating tracks...")
            use_stub = False

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=use_stub,
                                       stub_path=stub_path if use_stub else None)
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    
    # Verify camera movement stub data matches current frame count
    camera_stub_path = 'stubs/camera_movement_stub.pkl'
    use_camera_stub = True
    
    if Path(camera_stub_path).exists():
        try:
            with open(camera_stub_path, 'rb') as f:
                cached_camera_movement = pickle.load(f)
            # Check if cached camera movement has matching frame count
            if len(cached_camera_movement) != len(video_frames):
                print(f"Camera movement frame count mismatch: cached {len(cached_camera_movement)} frames vs current {len(video_frames)} frames")
                print("Regenerating camera movement...")
                use_camera_stub = False
        except Exception as e:
            print(f"Error reading camera movement stub: {e}. Regenerating...")
            use_camera_stub = False
    
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=use_camera_stub,
        stub_path=camera_stub_path if use_camera_stub else None
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Geometry layer: homography calibration and pixel → meter transform
    pixel_pts, world_pts = load_or_pick_points(
        video_frames[0],
        save_path='geometry/calibrations/wide_01.json'
    )
    H = build_homography(pixel_pts, world_pts)
    for object_name, object_tracks in tracks.items():
        for frame_num, frame_data in enumerate(object_tracks):
            for track_id, track_info in frame_data.items():
                position_adjusted = track_info.get('position_adjusted')
                if position_adjusted is None:
                    continue
                x_m, y_m = pixel_to_meters(position_adjusted[0], position_adjusted[1], H)
                tracks[object_name][frame_num][track_id]['position_transformed'] = [x_m, y_m]

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)  # unknown for first frame
    team_ball_control = np.array(team_ball_control)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.mp4')

    # Export match data JSON
    counter_path = Path('match_counter.txt')
    if counter_path.exists():
        match_num = int(counter_path.read_text().strip() or '0') + 1
    else:
        match_num = 1
    counter_path.write_text(str(match_num))
    match_id = f'match_{match_num:03d}'
    export_match_data(tracks, fps=30, match_id=match_id)

    # Generate heatmap and pass network visualizations
    assign_ball_touches(tracks)
    determine_ball_control(tracks)
    passes, turnovers = detect_passes(tracks)
    players = build_players(tracks, passes, turnovers)

    heatmap_by_team(players, team=1)
    heatmap_by_team(players, team=2)
    heatmap_for_team(players, team=1)
    heatmap_for_team(players, team=2)

    pass_network(players, passes, team=1)
    pass_network(players, passes, team=2)

if __name__ == '__main__':
    main()
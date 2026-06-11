# import necessary libraries
import json
import os
from pathlib import Path
import numpy as np

# generate a unique match ID by incrementing a counter stored in a file
def get_next_match_id():
    counter_file = 'match_counter.txt'
    
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            last_id = int(f.read().strip())
    else:
        last_id = 0
    
    new_id = last_id + 1
    
    with open(counter_file, 'w') as f:
        f.write(str(new_id))
    
    return f"match_{new_id:04d}"


# Function to determine which team has ball control based on player tracks
def determine_ball_control(tracks):    
    # intialize the previous player and team who has the ball to None
    previous_player_with_ball = None
    previous_team_with_ball = None

    #loop on frames and check which player is closest to the ball
    for frame_num, player_track in enumerate(tracks['players']):
        current_player_id = None
        current_team = None
        
        #loop on players in the current frame and check if they have the ball
        for player_id, track in player_track.items():
            if track.get('has_ball', False):
                current_player_id = player_id
                current_team = track.get('team', None)
                break
        # check ball loss by comparing current team with previous team
        if previous_player_with_ball is not None and current_team != previous_team_with_ball:
            frame = tracks['players'][frame_num - 1]
            previous = frame[previous_player_with_ball]
            previous['ball_loss'] = previous.get('ball_loss', 0) + 1

        # update the current player and team with ball control for the current frame
        previous_player_with_ball = current_player_id
        previous_team_with_ball = current_team


# Assign ball touches based on proximity to the ball
def assign_ball_touches(tracks, distance_threshold=120):
    # Check if the ball exists in this frame
    for frame_num, player_track in enumerate(tracks['players']):
        if not tracks['ball'][frame_num] or 1 not in tracks['ball'][frame_num]:
            continue
        # Calculate the center of the ball
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        ball_x = (ball_bbox[0] + ball_bbox[2]) / 2
        ball_y = (ball_bbox[1] + ball_bbox[3]) / 2

        # Check each player in the frame to see if they are close
        # enough to the ball to be considered as having touched it
        for player_id, track in player_track.items():
            player_bbox = track.get('bbox', [])
            if not player_bbox:
                continue

            player_x = (player_bbox[0] + player_bbox[2]) / 2
            player_y = (player_bbox[1] + player_bbox[3]) / 2

            distance = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)

            # If the player is within the distance threshold, we consider it a touch
            if distance < distance_threshold:
                tracks['players'][frame_num][player_id]['has_ball'] = True
                tracks['players'][frame_num][player_id]['touches'] = track.get('touches', 0) + 1
            else:
                tracks['players'][frame_num][player_id]['has_ball'] = False

# Detect passes and turnovers based on changes in ball control between players and teams
def detect_passes(tracks, distance_threshold=120, release_frames=3):
    # Intialize lists to store detected passes and turnovers
    passes = []
    turnovers = []
    # intialize variables to track the current ball owner, how many frames
    # the ball has been away from the owner, and the frame when the ball was released
    current_owner = None
    frames_away = 0
    release_frame = None
    passer = None

    # Loop through each frame and check for changes in ball control to identify passes and turnovers
    for frame_num, player_track in enumerate(tracks['players']):
        # Find who has the ball this frame
        frame_owner = None
        for player_id, track in player_track.items():
            if track.get('has_ball', False):
                frame_owner = {"player_id": player_id, "team": track.get("team")}
                break

        # If no one has the ball, check if we are in the middle of a potential pass (ball away from owner for a few frames)        
        if frame_owner is None:
            if current_owner is not None:
                frames_away += 1
                if frames_away >= release_frames and passer is None:
                    passer = current_owner
                    release_frame = frame_num
        else:
            # If the ball has a new owner, check if it's a pass or turnover
            if passer is not None:
                if frame_owner["player_id"] != passer["player_id"]:
                    if frame_owner["team"] == passer["team"]:
                        passes.append({
                            "frame_start": release_frame,
                            "frame_end": frame_num,
                            "passer_id": passer["player_id"],
                            "receiver_id": frame_owner["player_id"],
                            "team": passer["team"]
                        })
                    else:
                        turnovers.append({
                            "frame_start": release_frame,
                            "frame_end": frame_num,
                            "losing_player_id": passer["player_id"],
                            "gaining_player_id": frame_owner["player_id"],
                            "losing_team": passer["team"],
                            "gaining_team": frame_owner["team"]
                        })
                passer = None
                release_frame = None

            current_owner = frame_owner
            frames_away = 0

    return passes, turnovers
                
# Build player data dictionary by aggregating information across frames
def build_players(tracks, passes, turnovers):
    players = []
    # gather all unique player IDs across all frames
    all_player_ids = set()
    for frame_data in tracks['players']:
        all_player_ids.update(frame_data.keys())
    # Capture the last appearance of each player (has accumulated speed/distance totals)
    for track_id in all_player_ids:
        track_info = {}
        for frame_data in reversed(tracks['players']):
            if track_id in frame_data:
                track_info = frame_data[track_id]
                break
        # Calculate average position across all frames where the player appears
        # Uses position_transformed (real pitch meters) instead of raw pixel bbox
        positions = []
        for frame_num, frame_data in enumerate(tracks['players']):
            if track_id in frame_data:
                pos = frame_data[track_id].get('position_transformed')
                if pos is None:
                    pos = frame_data[track_id].get('position_adjusted')
                if pos is not None and isinstance(pos, (list, tuple)) and len(pos) == 2:
                    x, y = pos[0], pos[1]
                    if x is not None and y is not None:
                        positions.append({"frame": frame_num, "x": float(x), "y": float(y)})

        avg_x = round(np.mean([p["x"] for p in positions]), 2) if positions else 0
        avg_y = round(np.mean([p["y"] for p in positions]), 2) if positions else 0

        # Count passes made, passes received, and turnovers for this player
        passes_made     = sum(1 for p in passes    if p["passer_id"]        == track_id)
        passes_received = sum(1 for p in passes    if p["receiver_id"]      == track_id)
        turnovers_count = sum(1 for t in turnovers if t["losing_player_id"] == track_id)


        # Build player dictionary with aggregated information
        player_entry = {
            "player_id": track_id,
            "team": track_info.get("team", "unknown"),
            "ball_control": track_info.get("has_ball", False),
            "avg_speed": track_info.get("speed", 0),
            "distance_covered": track_info.get("distance", 0),
            "touches": track_info.get("touches", 0),
            "ball_loss": track_info.get("ball_loss", 0),
            "passes_made": passes_made,
            "passes_received": passes_received,
            "turnovers": turnovers_count,
            "avg_position": {"x": avg_x, "y": avg_y},
            "positions": positions
        }
        players.append(player_entry)

    return players

# Main function to export match data to JSON
def export_match_json(tracks, team_ball_control, team_names=["Team A", "Team B"]):
    # Calculate possession percentages based on ball control data
    team_ball_list = team_ball_control.tolist()
    team_1_possession = round(team_ball_list.count(1) / len(team_ball_list) * 100, 2)
    team_2_possession = round(100 - team_1_possession, 2)

    # Run touch assignment before building player data
    assign_ball_touches(tracks)

    # Run ball loss detection before building player data
    determine_ball_control(tracks)

    # Run pass and turnover detection before building player data
    passes, turnovers = detect_passes(tracks)


    # Build the match data dictionary to be exported as JSON
    match_data = {
        "match_id": get_next_match_id(),
        "team_names": team_names,
        "team_possession": {
            "team_1": team_1_possession,
            "team_2": team_2_possession
        },
        "passes": passes,
        "turnovers": turnovers,
        "player": build_players(tracks, passes, turnovers)
    }

    # Handle serialization of numpy data types when dumping to JSON
    def json_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Save the match data to a JSON file in the parent directory of this script
    output_path = Path(__file__).resolve().parent.parent / 'match_data.json'
    with open(output_path, 'w') as f:
        json.dump(match_data, f, indent=4, default=json_serializer)
    
    print(f"match_data.json saved to {output_path}")
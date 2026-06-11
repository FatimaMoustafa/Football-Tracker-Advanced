"""
Main Chunked Pipeline - Handles full video processing with memory optimization
"""

import glob
import os
import json
import pickle
import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
import traceback

# Import pipeline components
from pipeline.video_chunking import VideoChunker
from evaluation.eval_harness import TrackingEvaluator
from utils.bbox_utils import get_center_of_bbox, get_foot_position
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkedPipeline:
    """Memory-efficient pipeline processing video in chunks"""
    
    def __init__(self, model_path: str = 'models/best.pt', chunk_duration: int = 30):
        """Initialize pipeline with tracker and chunking parameters"""
        self.model_path = model_path
        self.chunk_duration = chunk_duration
        self.tracker = Tracker(model_path)
        self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()
        self.view_transformer = ViewTransformer()
        self.speed_estimator = SpeedAndDistance_Estimator()
    
    def process_video_chunked(self, 
                             video_path: str,
                             output_dir: str = 'output_videos',
                             use_stub: bool = False,
                             cleanup_temp_files: bool = False,
                             return_merged_tracks: bool = False) -> Dict:
        """
        Process video in chunks to avoid memory accumulation
        
        Args:
            video_path: Input video path
            output_dir: Output directory
            use_stub: Use cached stubs if available
            cleanup_temp_files: Remove temporary chunk files after success
            return_merged_tracks: Return merged tracks for evaluation if requested
            
        Returns:
            Dictionary with processing stats. If return_merged_tracks=True, includes merged tracks under 'tracks'.
        """
        logger.info("=" * 70)
        logger.info("CHUNKED VIDEO PROCESSING PIPELINE")
        logger.info("=" * 70)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        chunks_dir = os.path.join(output_dir, 'chunks')
        processed_dir = os.path.join(output_dir, 'processed_chunks')
        stats_dir = os.path.join(output_dir, 'chunk_stats')
        stubs_dir = os.path.join(output_dir, 'chunk_stubs')
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        Path(stats_dir).mkdir(parents=True, exist_ok=True)
        Path(stubs_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Split video into chunks
            logger.info(f"\n[STEP 1] Splitting {video_path} into {self.chunk_duration}s chunks...")
            chunk_paths = []
            if os.path.exists(chunks_dir) and any(p.endswith('.mp4') for p in os.listdir(chunks_dir)):
                chunk_paths = sorted(glob.glob(os.path.join(chunks_dir, 'chunk_*.mp4')))
                if chunk_paths and self._validate_chunk_set(chunk_paths, video_path):
                    logger.info(f"Found existing chunks in {chunks_dir}, reusing them")
                else:
                    logger.info(f"Existing chunk set is incomplete or invalid. Re-splitting video.")
                    chunk_paths = VideoChunker(chunk_duration=self.chunk_duration).split_video(video_path, chunks_dir)
            else:
                chunk_paths = VideoChunker(chunk_duration=self.chunk_duration).split_video(video_path, chunks_dir)
            logger.info(f"✓ Using {len(chunk_paths)} chunk(s)")
            
            # Step 2: Process each chunk
            logger.info(f"\n[STEP 2] Processing {len(chunk_paths)} chunks...")
            annotated_chunks = []
            chunk_stats_list = []
            merged_tracks = {
                'players': [],
                'referees': [],
                'ball': []
            } if return_merged_tracks else None
            
            for i, chunk_path in enumerate(chunk_paths):
                processed_chunk_path = os.path.join(processed_dir, f"chunk_{i:04d}_annotated.mp4")
                chunk_stats_path = os.path.join(stats_dir, f"chunk_{i:04d}_stats.json")
                chunk_stub_path = os.path.join(stubs_dir, f"chunk_{i:04d}_tracks.pkl")
                
                skip_chunk = os.path.exists(processed_chunk_path) and os.path.exists(chunk_stats_path)
                if skip_chunk:
                    logger.info(f"Skipping chunk {i+1}/{len(chunk_paths)}; already processed")
                    with open(chunk_stats_path, 'r') as f:
                        chunk_stats = json.load(f)
                    chunk_stats_list.append(chunk_stats)
                    annotated_chunks.append(processed_chunk_path)
                    if return_merged_tracks:
                        if os.path.exists(chunk_stub_path):
                            with open(chunk_stub_path, 'rb') as f:
                                chunk_tracks = pickle.load(f)
                            self._append_chunk_tracks(merged_tracks, chunk_tracks)
                        else:
                            logger.warning(f"Missing stub for chunk {i}, reprocessing to preserve merged tracks")
                            skip_chunk = False
                    if skip_chunk:
                        continue
                
                if not skip_chunk:
                    logger.info(f"\n--- Processing Chunk {i+1}/{len(chunk_paths)} ---")
                    
                    # Read chunk frames
                    cap = cv2.VideoCapture(chunk_path)
                    chunk_frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        chunk_frames.append(frame)
                    cap.release()
                
                if not chunk_frames:
                    logger.warning(f"No frames in chunk {i}")
                    continue
                
                logger.info(f"Processing {len(chunk_frames)} frames...")
                
                # Process chunk through tracker
                chunk_tracks = self.tracker.get_object_tracks(
                    chunk_frames,
                    read_from_stub=use_stub,
                    stub_path=chunk_stub_path
                )
                
                # Add positions and smooth IDs inside the chunk
                self.tracker.add_position_to_tracks(chunk_tracks)
                self._smooth_chunk_track_ids(chunk_tracks)
                if return_merged_tracks and merged_tracks and merged_tracks['players']:
                    self._link_chunk_boundary_ids(merged_tracks, chunk_tracks)
                
                # Camera movement
                try:
                    camera_estimator = CameraMovementEstimator(chunk_frames[0])
                    camera_movement = camera_estimator.get_camera_movement(
                        chunk_frames,
                        read_from_stub=False
                    )
                    camera_estimator.add_adjust_positions_to_tracks(chunk_tracks, camera_movement)
                except Exception as e:
                    logger.warning(f"Error in camera movement estimation: {e}")
                    camera_movement = [(0, 0) for _ in chunk_frames]
                    camera_estimator = CameraMovementEstimator(chunk_frames[0])
                
                # View transformation
                try:
                    self.view_transformer.add_transformed_position_to_tracks(chunk_tracks)
                except Exception as e:
                    logger.warning(f"Error in view transformation: {e}")
                
                # Interpolate ball
                chunk_tracks["ball"] = self.tracker.interpolate_ball_positions(chunk_tracks["ball"])
                
                # Speed and distance
                try:
                    self.speed_estimator.add_speed_and_distance_to_tracks(chunk_tracks)
                except Exception as e:
                    logger.warning(f"Error calculating speed and distance: {e}")
                
                # Team assignment
                for frame_num, player_track in enumerate(chunk_tracks['players']):
                    for player_id, track in player_track.items():
                        if 'bbox' not in track or not track['bbox']:
                            continue
                        try:
                            team = self.team_assigner.get_player_team(
                                chunk_frames[frame_num],
                                track['bbox'],
                                player_id
                            )
                            chunk_tracks['players'][frame_num][player_id]['team'] = team
                            chunk_tracks['players'][frame_num][player_id]['team_color'] = \
                                self.team_assigner.team_colors[team]
                        except Exception as e:
                            logger.debug(f"Error assigning team for player {player_id}: {e}")
                            chunk_tracks['players'][frame_num][player_id]['team'] = 0
                            chunk_tracks['players'][frame_num][player_id]['team_color'] = (0, 0, 255)
                
                # Ball assignment
                team_ball_control = []
                for frame_num, player_track in enumerate(chunk_tracks['players']):
                    ball_data = chunk_tracks['ball'][frame_num].get(1)
                    if ball_data is None:
                        assigned_player = -1
                    else:
                        ball_bbox = ball_data['bbox']
                        assigned_player = self.player_assigner.assign_ball_to_player(
                            player_track, ball_bbox
                        )

                    if assigned_player != -1:
                        chunk_tracks['players'][frame_num][assigned_player]['has_ball'] = True
                        team_ball_control.append(
                            chunk_tracks['players'][frame_num][assigned_player]['team']
                        )
                    else:
                        if len(team_ball_control) > 0:
                            team_ball_control.append(team_ball_control[-1])
                        else:
                            team_ball_control.append(0)
                
                team_ball_control = np.array(team_ball_control)
                
                # Free chunk_frames memory before drawing
                del chunk_frames
                
                # Draw annotations frame by frame to save memory
                self._draw_and_save_chunk(
                    chunk_path, processed_chunk_path, chunk_tracks, 
                    camera_estimator, camera_movement, team_ball_control
                )
                
                # Save stats
                os.makedirs(os.path.dirname(chunk_stats_path), exist_ok=True)
                chunk_stats = TrackingEvaluator().evaluate_tracks(chunk_tracks)
                with open(chunk_stats_path, 'w') as f:
                    json.dump(chunk_stats, f, indent=2, default=str)
                annotated_chunks.append(processed_chunk_path)
                
                # Collect results
                chunk_stats_list.append(chunk_stats)
                if return_merged_tracks:
                    self._append_chunk_tracks(merged_tracks, chunk_tracks)
                
                logger.info(f"✓ Chunk {i+1} processed: "
                           f"{len(chunk_tracks['players'])} frames, "
                           f"{len(chunk_tracks['players'][-1]) if chunk_tracks['players'] else 0} players")
                
                # Memory cleanup
                del chunk_tracks, camera_estimator
            
            # Step 3: Merge stats
            logger.info(f"\n[STEP 3] Merging statistics from {len(chunk_stats_list)} chunks...")
            merged_stats = self._merge_chunk_metrics(chunk_stats_list)
            if return_merged_tracks:
                merged_stats['tracks'] = merged_tracks
            
            # Step 4: Save outputs
            logger.info(f"\n[STEP 4] Saving outputs...")
            output_video_path = os.path.join(output_dir, 'output_video_chunked.mp4')
            self._concatenate_videos(annotated_chunks, output_video_path)
            
            # Save stats
            stats_path = os.path.join(output_dir, 'processing_stats.json')
            stats_to_save = merged_stats.copy()
            if return_merged_tracks:
                stats_to_save.pop('tracks', None)
            with open(stats_path, 'w') as f:
                json.dump(stats_to_save, f, indent=2, default=str)

            if return_merged_tracks and merged_tracks is not None:
                tracks_path = os.path.join(output_dir, 'merged_tracks.pkl')
                with open(tracks_path, 'wb') as f:
                    pickle.dump(merged_tracks, f)
                logger.info(f"Saved merged tracks to: {tracks_path}")
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ CHUNKED PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"Output video: {output_video_path}")
            logger.info(f"Stats file: {stats_path}")
            logger.info(f"Total chunks processed: {len(annotated_chunks)}")
            logger.info("=" * 70 + "\n")
            
            return merged_stats
            
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user. Partial results are preserved for resume.")
            raise
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            traceback.print_exc()
            raise
        finally:
            if cleanup_temp_files and os.path.exists(chunks_dir):
                logger.info(f"Cleaning up chunk files...")
                import shutil
                shutil.rmtree(chunks_dir, ignore_errors=True)
            if cleanup_temp_files and os.path.exists(processed_dir):
                logger.info(f"Cleaning up processed chunk outputs...")
                import shutil
                shutil.rmtree(processed_dir, ignore_errors=True)
            if cleanup_temp_files and os.path.exists(stats_dir):
                logger.info(f"Cleaning up chunk stats...")
                import shutil
                shutil.rmtree(stats_dir, ignore_errors=True)

    def _draw_and_save_chunk(self, chunk_path: str, output_path: str, 
                            chunk_tracks: Dict, camera_estimator, camera_movement: Dict, team_ball_control: np.ndarray):
        """Draw annotations on chunk frames one by one and save to video."""
        cap = cv2.VideoCapture(chunk_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw player tracks
            if frame_num < len(chunk_tracks['players']):
                player_dict = chunk_tracks['players'][frame_num]
                for track_id, player in player_dict.items():
                    if 'bbox' not in player or not player['bbox']:
                        continue
                    try:
                        color = player.get("team_color", (0, 0, 255))
                        frame = self.tracker.draw_ellipse(frame, player["bbox"], color, track_id)
                        if player.get('has_ball', False):
                            frame = self.tracker.draw_traingle(frame, player["bbox"], (0, 0, 255))
                    except Exception as e:
                        logger.debug(f"Error drawing player {track_id}: {e}")
                        continue
            
            # Draw ball
            if frame_num < len(chunk_tracks['ball']):
                ball_dict = chunk_tracks['ball'][frame_num]
                for track_id, ball in ball_dict.items():
                    if 'bbox' not in ball or not ball['bbox']:
                        continue
                    try:
                        bbox = ball["bbox"]
                        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                            continue
                        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in bbox):
                            continue
                        frame = self.tracker.draw_traingle(frame, bbox, (0, 255, 0))
                    except Exception as e:
                        logger.debug(f"Error drawing ball {track_id}: {e}")
                        continue
            
            # Draw team ball control
            frame = self.tracker.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            # Draw camera movement (single frame)
            frame = self._draw_camera_movement_single(frame, camera_movement, frame_num)
            
            # Draw speed and distance (single frame)
            frame = self._draw_speed_distance_single(frame, chunk_tracks, frame_num)
            
            writer.write(frame)
            frame_num += 1
        
        cap.release()
        writer.release()

    def _concatenate_videos(self, chunk_video_paths: List[str], output_path: str):
        """Concatenate processed chunk videos into a final output."""
        if not chunk_video_paths:
            raise ValueError("No processed chunk videos available to concatenate.")

        first_cap = cv2.VideoCapture(chunk_video_paths[0])
        width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = first_cap.get(cv2.CAP_PROP_FPS) or 24.0
        first_cap.release()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for video_path in chunk_video_paths:
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            cap.release()
        writer.release()

    def _append_chunk_tracks(self, merged_tracks: Dict, chunk_tracks: Dict):
        """Append chunked tracks into the merged track structure."""
        for key in ('players', 'referees', 'ball'):
            merged_tracks[key].extend(chunk_tracks.get(key, []))

    def _smooth_chunk_track_ids(self, chunk_tracks: Dict, max_dist: float = 50.0):
        """Smooth track IDs across adjacent frames within a chunk."""
        players = chunk_tracks.get('players', [])
        if not players or len(players) < 2:
            return

        for i in range(1, len(players)):
            prev_frame = players[i - 1]
            curr_frame = players[i]
            unmatched_prev = {pid: info for pid, info in prev_frame.items() if pid not in curr_frame}
            unmatched_curr = {cid: info for cid, info in curr_frame.items() if cid not in prev_frame}

            if not unmatched_prev or not unmatched_curr:
                continue

            prev_centers = {
                pid: get_center_of_bbox(info['bbox'])
                for pid, info in unmatched_prev.items()
                if 'bbox' in info and isinstance(info['bbox'], (list, tuple)) and len(info['bbox']) == 4
            }
            curr_centers = {
                cid: get_center_of_bbox(info['bbox'])
                for cid, info in unmatched_curr.items()
                if 'bbox' in info and isinstance(info['bbox'], (list, tuple)) and len(info['bbox']) == 4
            }

            if not prev_centers or not curr_centers:
                continue

            mapping = {}
            used_prev = set()
            used_curr = set()

            distances = []
            for prev_id, prev_center in prev_centers.items():
                for curr_id, curr_center in curr_centers.items():
                    dist = np.linalg.norm(np.array(prev_center) - np.array(curr_center))
                    distances.append((dist, prev_id, curr_id))

            distances.sort(key=lambda x: x[0])
            for dist, prev_id, curr_id in distances:
                if dist >= max_dist:
                    break
                if prev_id in used_prev or curr_id in used_curr:
                    continue
                mapping[curr_id] = prev_id
                used_prev.add(prev_id)
                used_curr.add(curr_id)

            if not mapping:
                continue

            self._remap_track_ids(players, i, mapping)

    def _link_chunk_boundary_ids(self, merged_tracks: Dict, chunk_tracks: Dict, max_dist: float = 80.0):
        """Link track IDs across chunk boundaries based on end-of-previous vs start-of-current positions."""
        if not merged_tracks or not merged_tracks.get('players') or not chunk_tracks.get('players'):
            return

        prev_frame = merged_tracks['players'][-1]
        curr_frame = chunk_tracks['players'][0]
        if not prev_frame or not curr_frame:
            return

        prev_positions = {
            pid: track_info.get('position')
            for pid, track_info in prev_frame.items()
            if track_info.get('position') is not None
        }
        curr_positions = {
            cid: track_info.get('position')
            for cid, track_info in curr_frame.items()
            if track_info.get('position') is not None
        }

        distances = []
        for prev_id, prev_pos in prev_positions.items():
            for curr_id, curr_pos in curr_positions.items():
                dist = np.linalg.norm(np.array(prev_pos) - np.array(curr_pos))
                distances.append((dist, prev_id, curr_id))

        distances.sort(key=lambda x: x[0])
        mapping = {}
        used_prev = set()
        used_curr = set()

        for dist, prev_id, curr_id in distances:
            if dist >= max_dist:
                break
            if prev_id in used_prev or curr_id in used_curr:
                continue
            if prev_id == curr_id:
                continue
            mapping[curr_id] = prev_id
            used_prev.add(prev_id)
            used_curr.add(curr_id)

        if mapping:
            self._remap_track_ids(chunk_tracks['players'], 0, mapping)

    def _remap_track_ids(self, player_frames: List[Dict], start_frame: int, mapping: Dict[int, int]):
        """Remap player IDs from a starting frame onward."""
        for frame in player_frames[start_frame:]:
            for old_id, new_id in list(mapping.items()):
                if old_id in frame:
                    if new_id in frame:
                        # Keep existing new_id if already present; remove duplicate old_id
                        del frame[old_id]
                    else:
                        frame[new_id] = frame.pop(old_id)

    def _validate_chunk_set(self, chunk_paths: List[str], video_path: str) -> bool:
        """Validate whether existing chunk files match the expected video split."""
        try:
            duration = VideoChunker(chunk_duration=self.chunk_duration)._get_video_duration(video_path)
            expected_chunks = (int(duration) // self.chunk_duration) + 1
            if len(chunk_paths) != expected_chunks:
                return False
            expected_names = [f"chunk_{i:04d}.mp4" for i in range(expected_chunks)]
            return [os.path.basename(p) for p in chunk_paths] == expected_names
        except Exception:
            return False

    def _draw_camera_movement_single(self, frame, camera_movement, frame_num):
        """Draw camera movement on a single frame."""
        if frame_num >= len(camera_movement):
            return frame
        
        frame = frame.copy()
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        x_movement, y_movement = camera_movement[frame_num]
        cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return frame
    
    def _draw_speed_distance_single(self, frame, tracks, frame_num):
        """Draw speed and distance on a single frame."""
        frame = frame.copy()
        
        # Draw speed and distance for players
        if 'players' in tracks and frame_num < len(tracks['players']):
            for _, track_info in tracks['players'][frame_num].items():
                if "speed" in track_info:
                    speed = track_info.get('speed', None)
                    distance = track_info.get('distance', None)
                    if speed is None or distance is None:
                        continue
                    
                    bbox = track_info['bbox']
                    position = self._get_foot_position(bbox)
                    position = list(position)
                    position[1] += 40
                    position = tuple(map(int, position))
                    
                    cv2.putText(frame, f"{speed:.2f} km/h", position, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def _get_foot_position(self, bbox):
        """Get foot position from bbox."""
        x1, y1, x2, y2 = bbox
        return (x1 + (x2 - x1) / 2, y2)

    def _merge_chunk_metrics(self, chunk_stats_list: List[Dict]) -> Dict:
        """Merge metrics from all chunks"""
        merged = {
            'total_chunks': len(chunk_stats_list),
            'chunks': chunk_stats_list,
            'combined_score': 0
        }

        if chunk_stats_list:
            scores = []
            for stats in chunk_stats_list:
                if 'summary' in stats:
                    rating = self._get_rating_score(stats)
                    scores.append(rating)

            if scores:
                merged['combined_score'] = float(np.mean(scores))
                merged['rating'] = self._score_to_rating(merged['combined_score'])

        return merged

    def _get_rating_score(self, metrics: Dict) -> float:
        """Convert metrics to 0-100 score"""
        score = 50

        if 'id_switches' in metrics:
            switches_score = max(0, 100 - metrics['id_switches'].get('avg_per_frame', 0) * 1000)
            score += switches_score * 0.3

        if 'fragmentation' in metrics:
            frag_score = (1 - metrics['fragmentation'].get('fragmentation_score', 0)) * 100
            score += frag_score * 0.3

        if 'speed_plausibility' in metrics:
            speed_score = (1 - metrics['speed_plausibility'].get('implausibility_rate', 0)) * 100
            score += speed_score * 0.2

        if 'track_consistency' in metrics:
            consistency_score = metrics['track_consistency'].get('consistency_score', 0) * 100
            score += consistency_score * 0.2

        return score

    def _score_to_rating(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 85:
            return 'A (Excellent)'
        elif score >= 70:
            return 'B (Good)'
        elif score >= 55:
            return 'C (Fair)'
        else:
            return 'F (Poor)'


def main():
    parser = argparse.ArgumentParser(description='Chunked Football Tracking Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='output_videos', help='Output directory')
    parser.add_argument('--chunk-duration', type=int, default=30, help='Chunk duration in seconds')
    parser.add_argument('--use-stub', action='store_true', help='Use cached stubs')
    parser.add_argument('--model', default='models/best.pt', help='Model path')
    
    args = parser.parse_args()
    
    pipeline = ChunkedPipeline(
        model_path=args.model,
        chunk_duration=args.chunk_duration
    )
    
    stats = pipeline.process_video_chunked(
        video_path=args.input,
        output_dir=args.output,
        use_stub=args.use_stub
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info(json.dumps(stats, indent=2, default=str))
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

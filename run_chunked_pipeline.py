"""
Main Chunked Pipeline - Handles full video processing with memory optimization
"""

import glob
import os
import json
import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging
import traceback

# Import pipeline components
from pipeline.video_chunking import chunk_and_process_video, VideoChunker, JSONMerger, ChunkProcessor
from evaluation.eval_harness import TrackingEvaluator, evaluate_test_clip
from utils import read_video, save_video
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
    
    def __init__(self, model_path: str = 'models/best.pt', chunk_duration: int = 60):
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
                             cleanup_temp_files: bool = False) -> Dict:
        """
        Process video in chunks to avoid memory accumulation
        
        Args:
            video_path: Input video path
            output_dir: Output directory
            use_stub: Use cached stubs if available
            cleanup_temp_files: Remove temporary chunk files after success
            
        Returns:
            Dictionary with processing stats
        """
        logger.info("=" * 70)
        logger.info("CHUNKED VIDEO PROCESSING PIPELINE")
        logger.info("=" * 70)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        chunks_dir = os.path.join(output_dir, 'chunks')
        processed_dir = os.path.join(output_dir, 'processed_chunks')
        stats_dir = os.path.join(output_dir, 'chunk_stats')
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        Path(stats_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Split video into chunks
            logger.info(f"\n[STEP 1] Splitting {video_path} into {self.chunk_duration}s chunks...")
            if os.path.exists(chunks_dir) and any(p.endswith('.mp4') for p in os.listdir(chunks_dir)):
                chunk_paths = sorted(glob.glob(os.path.join(chunks_dir, 'chunk_*.mp4')))
                if chunk_paths:
                    logger.info(f"Found existing chunks in {chunks_dir}, reusing them")
                else:
                    chunk_paths = VideoChunker(chunk_duration=self.chunk_duration).split_video(video_path, chunks_dir)
            else:
                chunk_paths = VideoChunker(chunk_duration=self.chunk_duration).split_video(video_path, chunks_dir)
            logger.info(f"✓ Using {len(chunk_paths)} chunk(s)")
            
            # Step 2: Process each chunk
            logger.info(f"\n[STEP 2] Processing {len(chunk_paths)} chunks...")
            annotated_chunks = []
            chunk_stats_list = []
            
            for i, chunk_path in enumerate(chunk_paths):
                processed_chunk_path = os.path.join(processed_dir, f"chunk_{i:04d}_annotated.mp4")
                chunk_stats_path = os.path.join(stats_dir, f"chunk_{i:04d}_stats.json")
                
                if os.path.exists(processed_chunk_path) and os.path.exists(chunk_stats_path):
                    logger.info(f"Skipping chunk {i+1}/{len(chunk_paths)}; already processed")
                    with open(chunk_stats_path, 'r') as f:
                        chunk_stats_list.append(json.load(f))
                    annotated_chunks.append(processed_chunk_path)
                    continue
                
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
                    stub_path=f'stubs/chunk_{i}_tracks.pkl' if use_stub else None
                )
                
                # Add positions
                self.tracker.add_position_to_tracks(chunk_tracks)
                
                # Camera movement
                camera_estimator = CameraMovementEstimator(chunk_frames[0])
                camera_movement = camera_estimator.get_camera_movement(
                    chunk_frames,
                    read_from_stub=False
                )
                camera_estimator.add_adjust_positions_to_tracks(chunk_tracks, camera_movement)
                
                # View transformation
                self.view_transformer.add_transformed_position_to_tracks(chunk_tracks)
                
                # Interpolate ball
                chunk_tracks["ball"] = self.tracker.interpolate_ball_positions(chunk_tracks["ball"])
                
                # Speed and distance
                self.speed_estimator.add_speed_and_distance_to_tracks(chunk_tracks)
                
                # Team assignment
                for frame_num, player_track in enumerate(chunk_tracks['players']):
                    for player_id, track in player_track.items():
                        team = self.team_assigner.get_player_team(
                            chunk_frames[frame_num],
                            track['bbox'],
                            player_id
                        )
                        chunk_tracks['players'][frame_num][player_id]['team'] = team
                        chunk_tracks['players'][frame_num][player_id]['team_color'] = \
                            self.team_assigner.team_colors[team]
                
                # Ball assignment
                team_ball_control = []
                for frame_num, player_track in enumerate(chunk_tracks['players']):
                    ball_bbox = chunk_tracks['ball'][frame_num][1]['bbox']
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
                
                # Draw annotations
                output_frames = self.tracker.draw_annotations(chunk_frames, chunk_tracks, team_ball_control)
                output_frames = camera_estimator.draw_camera_movement(output_frames, camera_movement)
                self.speed_estimator.draw_speed_and_distance(output_frames, chunk_tracks)
                
                # Save annotated chunk and stats for resume support
                save_video(output_frames, processed_chunk_path)
                with open(chunk_stats_path, 'w') as f:
                    json.dump(TrackingEvaluator().evaluate_tracks(chunk_tracks), f, indent=2, default=str)
                annotated_chunks.append(processed_chunk_path)
                
                # Collect results
                chunk_stats_list.append(json.loads(open(chunk_stats_path, 'r').read()))
                
                logger.info(f"✓ Chunk {i+1} processed: "
                           f"{len(chunk_frames)} frames, "
                           f"{len(chunk_tracks['players'][-1]) if chunk_tracks['players'] else 0} players")
                
                # Memory cleanup
                del chunk_frames, chunk_tracks, camera_estimator, output_frames
            
            # Step 3: Merge stats
            logger.info(f"\n[STEP 3] Merging statistics from {len(chunk_stats_list)} chunks...")
            merged_stats = self._merge_chunk_metrics(chunk_stats_list)
            
            # Step 4: Save outputs
            logger.info(f"\n[STEP 4] Saving outputs...")
            output_video_path = os.path.join(output_dir, 'output_video_chunked.mp4')
            self._concatenate_videos(annotated_chunks, output_video_path)
            
            # Save stats
            stats_path = os.path.join(output_dir, 'processing_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(merged_stats, f, indent=2, default=str)
            
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
    parser.add_argument('--chunk-duration', type=int, default=60, help='Chunk duration in seconds')
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

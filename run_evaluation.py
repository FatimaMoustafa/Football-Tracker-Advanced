"""
Evaluation Harness Script - Run tracking evaluation on test clips
Measures: ID switches, fragmentation, speed plausibility
"""

import glob
import os
import json
import pickle
import argparse
import cv2
from pathlib import Path
import logging
import traceback

from run_chunked_pipeline import ChunkedPipeline
from evaluation.eval_harness import TrackingEvaluator
from evaluation.chunk_boundary_validator import validate_chunk_boundary_continuity
from utils import read_video
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


class EvaluationHarness:
    """Complete evaluation toolkit for tracking quality assessment"""
    
    def __init__(self, model_path: str = 'models/best.pt', chunk_duration: int = 30):
        """Initialize evaluation harness"""
        self.model_path = model_path
        self.chunk_duration = chunk_duration
        self.tracker = Tracker(model_path)
        self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()
        self.view_transformer = ViewTransformer()
        self.speed_estimator = SpeedAndDistance_Estimator()
        self.evaluator = None
    
    def run_full_pipeline(self, video_path: str) -> dict:
        """
        Run complete tracking pipeline on video
        
        Args:
            video_path: Path to test video
            
        Returns:
            Dictionary with tracks
        """
        logger.info(f"Loading video: {video_path}")
        frames = read_video(video_path)
        
        if not frames:
            raise ValueError(f"No frames loaded from {video_path}")
        
        logger.info(f"Loaded {len(frames)} frames from video")
        
        # Get tracks
        logger.info("Running object detection...")
        tracks = self.tracker.get_object_tracks(frames, read_from_stub=False)
        
        # Add positions
        self.tracker.add_position_to_tracks(tracks)
        
        # Camera movement
        logger.info("Estimating camera movement...")
        camera_estimator = CameraMovementEstimator(frames[0])
        camera_movement = camera_estimator.get_camera_movement(frames, read_from_stub=False)
        camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
        
        # View transformation
        logger.info("Applying perspective transformation...")
        self.view_transformer.add_transformed_position_to_tracks(tracks)
        
        # Ball interpolation
        tracks["ball"] = self.tracker.interpolate_ball_positions(tracks["ball"])
        
        # Speed and distance
        logger.info("Calculating speed and distance...")
        self.speed_estimator.add_speed_and_distance_to_tracks(tracks)
        
        # Team assignment
        logger.info("Assigning player teams...")
        import numpy as np
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = self.team_assigner.get_player_team(
                    frames[frame_num],
                    track['bbox'],
                    player_id
                )
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = \
                    self.team_assigner.team_colors[team]
        
        # Ball assignment
        logger.info("Assigning ball possession...")
        team_ball_control = []
        for frame_num, player_track in enumerate(tracks['players']):
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = self.player_assigner.assign_ball_to_player(
                player_track, ball_bbox
            )
            
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(
                    tracks['players'][frame_num][assigned_player]['team']
                )
            else:
                if len(team_ball_control) > 0:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(0)
        
        return tracks
    
    def run_chunked_pipeline(self, video_path: str, output_dir: str, chunk_duration: int = None) -> dict:
        """Run chunked tracking pipeline and return merged tracks."""
        pipeline_output_dir = os.path.join(output_dir, 'chunked_processing')
        pipeline = ChunkedPipeline(
            model_path=self.model_path,
            chunk_duration=chunk_duration or self.chunk_duration
        )
        result = pipeline.process_video_chunked(
            video_path=video_path,
            output_dir=pipeline_output_dir,
            use_stub=False,
            cleanup_temp_files=False,
            return_merged_tracks=True
        )
        tracks = result.get('tracks', {})
        if not tracks:
            raise RuntimeError('Chunked processing did not produce merged tracks.')
        return tracks

    def evaluate_video(self, video_path: str, output_dir: str = 'evaluation_results', use_chunked: bool = True, chunk_duration: int = None, existing_processing_dir: str = None, existing_tracks: dict = None) -> dict:
        """
        Complete evaluation pipeline
        
        Args:
            video_path: Path to test video
            output_dir: Directory to save evaluation results
            use_chunked: Whether to run evaluation using chunked processing
            chunk_duration: Duration of each chunk for streaming evaluation
            existing_processing_dir: Reuse previously processed chunked data from this directory
            existing_tracks: Use already merged tracks directly without rerunning processing
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=" * 70)
        logger.info("TRACKING EVALUATION HARNESS")
        logger.info("=" * 70)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Run tracking pipeline
            logger.info("\n[STEP 1] Running tracking pipeline...")
            if use_chunked:
                if existing_tracks is not None:
                    logger.info("Reusing merged tracks from prior processing for evaluation.")
                    tracks = existing_tracks
                elif existing_processing_dir is not None:
                    logger.info("Loading merged tracks from existing processing directory for evaluation.")
                    tracks = self._load_tracks_from_processing_dir(existing_processing_dir)
                else:
                    tracks = self.run_chunked_pipeline(
                        video_path=video_path,
                        output_dir=output_dir,
                        chunk_duration=chunk_duration
                    )
            else:
                tracks = self.run_full_pipeline(video_path)
            
            # Step 2: Evaluate tracks
            logger.info("\n[STEP 2] Evaluating tracking quality...")
            self.evaluator = TrackingEvaluator(fps=24)
            metrics = self.evaluator.evaluate_tracks(tracks)
            
            # Step 2b: Validate chunk boundary continuity (if using chunked processing)
            boundary_validation = None
            if use_chunked:
                logger.info("\n[STEP 2b] Validating chunk boundary ID continuity...")
                boundary_validation = validate_chunk_boundary_continuity(
                    tracks,
                    output_json=os.path.join(output_dir, 'boundary_continuity_metrics.json')
                )
                metrics['boundary_continuity'] = boundary_validation
            
            # Step 3: Generate report
            logger.info("\n[STEP 3] Generating evaluation report...")
            report = self.evaluator.print_report(verbose=True)
            
            # Step 4: Save results
            logger.info("\n[STEP 4] Saving evaluation results...")
            
            # Save metrics JSON
            metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
            self.evaluator.export_metrics_json(metrics_path)
            
            # Save report text
            report_path = os.path.join(output_dir, 'evaluation_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Save HTML report
            html_report = self._generate_html_report(metrics)
            html_path = os.path.join(output_dir, 'evaluation_report.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ EVALUATION COMPLETED")
            logger.info("=" * 70)
            logger.info(f"Metrics saved to: {metrics_path}")
            logger.info(f"Report saved to: {report_path}")
            logger.info(f"HTML report saved to: {html_path}")
            logger.info("=" * 70 + "\n")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            traceback.print_exc()
            raise
    
    def _generate_html_report(self, metrics: dict) -> str:
        """Generate HTML evaluation report"""
        summary = metrics['summary']
        id_switches = metrics['id_switches']
        fragmentation = metrics['fragmentation']
        speed = metrics['speed_plausibility']
        consistency = metrics['track_consistency']
        boundary_validation = metrics.get('boundary_continuity', {})
        
        rating = self.evaluator._calculate_overall_rating()
        
        # Build boundary continuity section if available
        boundary_section = ""
        if boundary_validation:
            continuity_score = boundary_validation.get('continuity_score', 0)
            num_boundaries = boundary_validation.get('num_boundaries', 0)
            boundary_section = f"""
            <h2> Chunk Boundary Continuity</h2>
            <div>
                <div class="metric">
                    <div class="metric-label">Continuity Score</div>
                    <div class="metric-value {'' if continuity_score >= 0.65 else 'warning'}">
                        {continuity_score:.3f}
                    </div>
                    <small>(0=poor, 1=perfect)</small>
                </div>
                <div class="metric">
                    <div class="metric-label">Boundaries Detected</div>
                    <div class="metric-value">{num_boundaries}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">ID Retention Rate</div>
                    <div class="metric-value">{boundary_validation.get('id_retention_rate', 0):.3f}</div>
                </div>
            </div>
            <p style="margin-left: 10px; font-style: italic; color: #666;">
                {boundary_validation.get('recommendation', 'No recommendation')}
            </p>
            """
        
        html = f"""
        <html>
        <head>
            <title>Tracking Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 5px; }}
                h1 {{ color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
                h2 {{ color: #0066cc; margin-top: 30px; border-left: 5px solid #0066cc; padding-left: 10px; }}
                .metric {{ display: inline-block; background: #f0f0f0; padding: 15px; margin: 10px; 
                           border-radius: 5px; border-left: 5px solid #0066cc; min-width: 250px; }}
                .metric-label {{ font-weight: bold; color: #0066cc; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
                .rating {{ padding: 20px; background: #e8f4f8; border-radius: 5px; margin: 20px 0; }}
                .rating-score {{ font-size: 32px; font-weight: bold; color: #0066cc; }}
                .rating-status {{ font-size: 18px; color: #333; margin-top: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #0066cc; color: white; }}
                .warning {{ color: #ff6600; font-weight: bold; }}
                .success {{ color: #00cc00; font-weight: bold; }}
            </style>
        </head>
        <body>
        <div class="container">
            <h1>🎯 Tracking Quality Evaluation Report</h1>
            
            <div class="rating">
                <div class="rating-score">{rating['rating']}</div>
                <div class="rating-status">{rating['status']}</div>
                <div>Overall Score: {rating['percentage']:.1f}%</div>
            </div>
            
            <h2>📊 Summary Statistics</h2>
            <div>
                <div class="metric">
                    <div class="metric-label">Total Frames</div>
                    <div class="metric-value">{summary['total_frames']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Unique Players</div>
                    <div class="metric-value">{summary['unique_players']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Players/Frame</div>
                    <div class="metric-value">{summary['avg_players_per_frame']:.1f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Track Duration</div>
                    <div class="metric-value">{summary['avg_track_duration_frames']:.1f}f</div>
                </div>
            </div>
            
            <h2>🔄 ID Switches</h2>
            <div>
                <div class="metric">
                    <div class="metric-label">Total Switches</div>
                    <div class="metric-value {'' if id_switches['total_switches'] < 10 else 'warning'}">
                        {id_switches['total_switches']}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg per Frame</div>
                    <div class="metric-value">{id_switches['avg_per_frame']:.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Peak Frame</div>
                    <div class="metric-value">{id_switches['peak_switches_frame']}</div>
                </div>
            </div>
            
            <h2>📍 Fragmentation</h2>
            <div>
                <div class="metric">
                    <div class="metric-label">Fragmentation Score</div>
                    <div class="metric-value">{fragmentation['fragmentation_score']:.3f}</div>
                    <small>(0=good, 1=bad)</small>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Track Length</div>
                    <div class="metric-value">{fragmentation['avg_track_length_frames']:.1f}f</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Tracks</div>
                    <div class="metric-value">{fragmentation['num_tracks']}</div>
                </div>
            </div>
            
            <h2>⚡ Speed Plausibility</h2>
            <div>
                <div class="metric">
                    <div class="metric-label">Implausible Movements</div>
                    <div class="metric-value {'' if speed['implausible_movements'] < 5 else 'warning'}">
                        {speed['implausible_movements']}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">Implausibility Rate</div>
                    <div class="metric-value">{speed['implausibility_rate']:.4f}</div>
                </div>
            </div>
            
            <h2>✓ Consistency</h2>
            <div>
                <div class="metric">
                    <div class="metric-label">Consistency Score</div>
                    <div class="metric-value {'' if consistency['consistency_score'] > 0.7 else 'warning'}">
                        {consistency['consistency_score']:.3f}
                    </div>
                    <small>(0=bad, 1=good)</small>
                </div>
                <div class="metric">
                    <div class="metric-label">Std Dev</div>
                    <div class="metric-value">{consistency['std_players_per_frame']:.2f}</div>
                </div>
            </div>
            
            {boundary_section}
            
            <h2>📈 Component Scores</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Score (%)</th>
                    <th>Weight</th>
                </tr>
                <tr>
                    <td>ID Switches</td>
                    <td>{rating['components']['id_switches']:.1f}%</td>
                    <td>30%</td>
                </tr>
                <tr>
                    <td>Fragmentation</td>
                    <td>{rating['components']['fragmentation']:.1f}%</td>
                    <td>30%</td>
                </tr>
                <tr>
                    <td>Speed Plausibility</td>
                    <td>{rating['components']['speed_plausibility']:.1f}%</td>
                    <td>20%</td>
                </tr>
                <tr>
                    <td>Consistency</td>
                    <td>{rating['components']['consistency']:.1f}%</td>
                    <td>20%</td>
                </tr>
            </table>
            
            <p style="margin-top: 40px; color: #999; font-size: 12px;">
                Generated by Football Tracker Evaluation Harness
            </p>
        </div>
        </body>
        </html>
        """
        
        return html
    
    def _load_tracks_from_processing_dir(self, processing_dir: str) -> dict:
        """Load merged tracks from a previously completed chunked processing output."""
        merged_tracks_path = os.path.join(processing_dir, 'merged_tracks.pkl')
        if not os.path.exists(merged_tracks_path):
            raise FileNotFoundError(
                f"Merged track data not found in processing directory: {merged_tracks_path}"
            )

        with open(merged_tracks_path, 'rb') as f:
            tracks = pickle.load(f)

        if not tracks:
            raise ValueError("Loaded merged tracks file is empty.")

        return tracks


def main():
    parser = argparse.ArgumentParser(description='Tracking Evaluation Harness')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='evaluation_results', help='Output directory')
    parser.add_argument('--model', default='models/best.pt', help='Model path')
    parser.add_argument('--chunk-duration', type=int, default=30, help='Chunk duration in seconds for evaluation')
    parser.add_argument('--no-chunked', action='store_true', help='Disable chunked evaluation and use full video processing')
    
    args = parser.parse_args()
    
    harness = EvaluationHarness(model_path=args.model, chunk_duration=args.chunk_duration)
    metrics = harness.evaluate_video(
        args.input,
        output_dir=args.output,
        use_chunked=not args.no_chunked,
        chunk_duration=args.chunk_duration
    )
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(json.dumps(
        {
            'id_switches': metrics['id_switches']['total_switches'],
            'avg_per_frame': metrics['id_switches']['avg_per_frame'],
            'fragmentation': metrics['fragmentation']['fragmentation_score'],
            'implausible_moves': metrics['speed_plausibility']['implausible_movements']
        },
        indent=2
    ))
    print("=" * 70)


if __name__ == '__main__':
    main()

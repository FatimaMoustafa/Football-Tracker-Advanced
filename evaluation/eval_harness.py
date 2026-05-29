"""
Evaluation Harness for Tracking Quality Assessment
Measures: ID switch count, track fragmentation, speed plausibility
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrackingEvaluator:
    """Comprehensive tracking quality evaluation"""
    
    def __init__(self, fps: int = 24):
        """
        Initialize evaluator
        
        Args:
            fps: Frames per second of video
        """
        self.fps = fps
        self.metrics = {}
    
    def evaluate_tracks(self, tracks: Dict) -> Dict:
        """
        Evaluate tracking quality from track dictionary
        
        Args:
            tracks: Dictionary containing player tracks
                   Format: {object_type: {frame_num: {track_id: track_info}}}
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating tracking quality...")
        
        metrics = {
            'id_switches': self._calculate_id_switches(tracks),
            'fragmentation': self._calculate_fragmentation(tracks),
            'speed_plausibility': self._evaluate_speed_plausibility(tracks),
            'track_consistency': self._calculate_track_consistency(tracks),
            'summary': {}
        }
        
        # Summary statistics
        metrics['summary'] = {
            'total_frames': self._count_total_frames(tracks),
            'unique_players': self._count_unique_players(tracks),
            'avg_players_per_frame': self._calculate_avg_players_per_frame(tracks),
            'avg_track_duration_frames': self._calculate_avg_track_duration(tracks)
        }
        
        self.metrics = metrics
        return metrics
    
    def _calculate_id_switches(self, tracks: Dict) -> Dict:
        """
        Calculate ID switches (track breaks)
        
        ID switch occurs when same player is assigned different ID across frames
        """
        logger.info("Calculating ID switches...")
        
        if 'players' not in tracks:
            return {'total_switches': 0, 'per_frame': []}
        
        player_tracks = tracks['players']
        id_switches = []
        total_switches = 0
        
        # Track player positions to detect re-identification
        prev_positions = {}  # player_id -> position
        switches_per_frame = []
        
        for frame_num, frame_tracks in enumerate(player_tracks):
            frame_switches = 0
            curr_positions = {
                player_id: track_info.get('position', [0, 0])
                for player_id, track_info in frame_tracks.items()
            }

            if prev_positions:
                # Build all pairwise distances between previous and current IDs
                distances = []
                for prev_id, prev_pos in prev_positions.items():
                    for curr_id, curr_pos in curr_positions.items():
                        dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 +
                                       (curr_pos[1] - prev_pos[1])**2)
                        distances.append((prev_id, curr_id, dist))

                distances.sort(key=lambda x: x[2])
                used_prev = set()
                used_curr = set()
                for prev_id, curr_id, dist in distances:
                    if dist >= 50:
                        break
                    if prev_id in used_prev or curr_id in used_curr:
                        continue
                    used_prev.add(prev_id)
                    used_curr.add(curr_id)
                    if prev_id != curr_id:
                        frame_switches += 1
                        total_switches += 1

            switches_per_frame.append(frame_switches)
            prev_positions = curr_positions
        
        return {
            'total_switches': total_switches,
            'per_frame': switches_per_frame,
            'avg_per_frame': total_switches / len(player_tracks) if player_tracks else 0,
            'peak_switches_frame': max(enumerate(switches_per_frame), 
                                      key=lambda x: x[1])[0] if switches_per_frame else -1
        }
    
    def _calculate_fragmentation(self, tracks: Dict) -> Dict:
        """
        Calculate track fragmentation score
        
        High fragmentation = many short tracks (bad)
        Low fragmentation = few long tracks (good)
        """
        logger.info("Calculating track fragmentation...")
        
        if 'players' not in tracks:
            return {'fragmentation_score': 0, 'track_lengths': []}
        
        player_tracks = tracks['players']
        track_segments = []
        active_segments = {}

        for frame_tracks in player_tracks:
            current_ids = set(frame_tracks.keys())
            for player_id in current_ids:
                if player_id in active_segments:
                    active_segments[player_id] += 1
                else:
                    active_segments[player_id] = 1

            finished_ids = [pid for pid in active_segments if pid not in current_ids]
            for pid in finished_ids:
                track_segments.append(active_segments.pop(pid))

        track_segments.extend(active_segments.values())

        if not track_segments:
            return {'fragmentation_score': 0, 'track_lengths': []}

        lengths = list(track_segments)
        avg_length = np.mean(lengths)
        std_length = np.std(lengths) if len(lengths) > 1 else 0
        total_segment_frames = sum(lengths)
        long_segment_frame_threshold = 120  # 5 seconds at 24fps
        long_segment_frames = sum(length for length in lengths if length >= long_segment_frame_threshold)
        long_segment_ratio = long_segment_frames / max(total_segment_frames, 1)

        fragmentation_score = 1 - long_segment_ratio

        return {
            'fragmentation_score': fragmentation_score,
            'avg_track_length_frames': avg_length,
            'std_track_length': std_length,
            'num_tracks': len(lengths),
            'min_track_length': min(lengths),
            'max_track_length': max(lengths),
            'track_lengths': lengths,
            'long_track_frame_ratio': long_segment_ratio,
            'long_track_frame_threshold': long_segment_frame_threshold
        }
    
    def _evaluate_speed_plausibility(self, tracks: Dict) -> Dict:
        """
        Evaluate speed plausibility of player movements
        
        Detects unrealistic jumps (teleportation)
        Typical player max speed: ~12 m/s ≈ 400 pixels/s at typical field zoom
        """
        logger.info("Evaluating speed plausibility...")
        
        if 'players' not in tracks:
            return {'implausible_movements': 0, 'details': []}
        
        player_tracks = tracks['players']
        implausible_count = 0
        implausible_details = []
        
        # Max pixel distance per frame after compensation
        MAX_PIXELS_PER_FRAME = 240
        
        prev_positions = {}
        
        for frame_num, frame_tracks in enumerate(player_tracks):
            for player_id, track_info in frame_tracks.items():
                position = track_info.get('position_transformed')
                if position is None:
                    position = track_info.get('position_adjusted')
                if position is None:
                    position = track_info.get('position', [0, 0])
                
                if player_id in prev_positions and prev_positions[player_id] is not None:
                    prev_pos = prev_positions[player_id]
                    distance = np.sqrt((position[0] - prev_pos[0])**2 + 
                                    (position[1] - prev_pos[1])**2)
                    
                    if distance > MAX_PIXELS_PER_FRAME:
                        implausible_count += 1
                        implausible_details.append({
                            'frame': frame_num,
                            'player_id': player_id,
                            'distance_pixels': float(distance),
                            'from_position': prev_pos,
                            'to_position': position
                        })
                
                prev_positions[player_id] = position
        
        total_frames = len(player_tracks)
        
        return {
            'implausible_movements': implausible_count,
            'implausibility_rate': implausible_count / max(total_frames, 1),
            'max_pixels_per_frame': MAX_PIXELS_PER_FRAME,
            'violations': implausible_details[:20]
        }
    
    def _calculate_track_consistency(self, tracks: Dict) -> Dict:
        """
        Calculate tracking consistency metrics
        """
        logger.info("Calculating track consistency...")
        
        if 'players' not in tracks:
            return {'consistency_score': 0}
        
        player_tracks = tracks['players']
        
        # Count unique players per frame
        players_per_frame = [len(frame_tracks) for frame_tracks in player_tracks]
        
        if not players_per_frame:
            return {'consistency_score': 0}
        
        # Consistency: low variance in players per frame
        avg_players = np.mean(players_per_frame)
        std_players = np.std(players_per_frame)
        
        consistency_score = max(0, 1 - (std_players / (avg_players + 1e-6)))
        
        return {
            'consistency_score': consistency_score,
            'avg_players_per_frame': avg_players,
            'std_players_per_frame': std_players,
            'min_players_frame': min(players_per_frame) if players_per_frame else 0,
            'max_players_frame': max(players_per_frame) if players_per_frame else 0
        }
    
    def _count_total_frames(self, tracks: Dict) -> int:
        """Count total frames in tracks"""
        if 'players' in tracks:
            return len(tracks['players'])
        return 0
    
    def _count_unique_players(self, tracks: Dict) -> int:
        """Count unique player IDs in all frames"""
        if 'players' not in tracks:
            return 0
        
        unique_ids = set()
        for frame_tracks in tracks['players']:
            unique_ids.update(frame_tracks.keys())
        return len(unique_ids)
    
    def _calculate_avg_players_per_frame(self, tracks: Dict) -> float:
        """Calculate average players detected per frame"""
        if 'players' not in tracks:
            return 0
        
        player_tracks = tracks['players']
        if not player_tracks:
            return 0
        
        return sum(len(frame_tracks) for frame_tracks in player_tracks) / len(player_tracks)
    
    def _calculate_avg_track_duration(self, tracks: Dict) -> float:
        """Calculate average track duration in frames"""
        if 'players' not in tracks:
            return 0
        
        player_tracks = tracks['players']
        track_lengths = defaultdict(int)
        
        for frame_tracks in player_tracks:
            for player_id in frame_tracks:
                track_lengths[player_id] += 1
        
        if not track_lengths:
            return 0
        
        return np.mean(list(track_lengths.values()))
    
    def print_report(self, verbose: bool = True) -> str:
        """Generate and print evaluation report"""
        if not self.metrics:
            return "No metrics available. Run evaluate_tracks first."
        
        report = []
        report.append("\n" + "=" * 70)
        report.append("TRACKING EVALUATION REPORT")
        report.append("=" * 70)
        
        # Summary
        summary = self.metrics['summary']
        report.append("\n[SUMMARY]")
        report.append(f"  Total Frames: {summary['total_frames']}")
        report.append(f"  Unique Players: {summary['unique_players']}")
        report.append(f"  Avg Players/Frame: {summary['avg_players_per_frame']:.2f}")
        report.append(f"  Avg Track Duration: {summary['avg_track_duration_frames']:.1f} frames")
        
        # ID Switches
        id_switches = self.metrics['id_switches']
        report.append("\n[ID SWITCHES]")
        report.append(f"  Total Switches: {id_switches['total_switches']}")
        report.append(f"  Avg per Frame: {id_switches['avg_per_frame']:.4f}")
        report.append(f"  Peak Frame: {id_switches['peak_switches_frame']}")
        
        # Fragmentation
        frag = self.metrics['fragmentation']
        report.append("\n[TRACK FRAGMENTATION]")
        report.append(f"  Fragmentation Score: {frag['fragmentation_score']:.3f} (0=good, 1=bad)")
        report.append(f"  Avg Track Length: {frag['avg_track_length_frames']:.1f} frames")
        report.append(f"  Min Track Length: {frag['min_track_length']} frames")
        report.append(f"  Max Track Length: {frag['max_track_length']} frames")
        report.append(f"  Total Tracks: {frag['num_tracks']}")
        
        # Speed Plausibility
        speed = self.metrics['speed_plausibility']
        report.append("\n[SPEED PLAUSIBILITY]")
        report.append(f"  Implausible Movements: {speed['implausible_movements']}")
        report.append(f"  Implausibility Rate: {speed['implausibility_rate']:.4f}")
        report.append(f"  Max Pixels/Frame Threshold: {speed['max_pixels_per_frame']}")
        
        if verbose and speed['violations']:
            report.append(f"\n  Top violations:")
            for v in speed['violations'][:5]:
                report.append(f"    Frame {v['frame']}: Player {v['player_id']} "
                            f"jumped {v['distance_pixels']:.0f}px")
        
        # Consistency
        consistency = self.metrics['track_consistency']
        report.append("\n[CONSISTENCY]")
        report.append(f"  Consistency Score: {consistency['consistency_score']:.3f} (0=bad, 1=good)")
        report.append(f"  Avg Players/Frame: {consistency['avg_players_per_frame']:.2f}")
        report.append(f"  Std Dev: {consistency['std_players_per_frame']:.2f}")
        
        # Quality Rating
        report.append("\n[QUALITY RATING]")
        rating = self._calculate_overall_rating()
        report.append(f"  Overall: {rating['rating']} ({rating['percentage']:.1f}%)")
        report.append(f"  Status: {rating['status']}")
        
        report.append("\n" + "=" * 70)
        
        report_str = "\n".join(report)
        
        if verbose:
            logger.info(report_str)
        
        return report_str
    
    def _calculate_overall_rating(self) -> Dict:
        """Calculate overall tracking quality rating"""
        if not self.metrics:
            return {'rating': 'N/A', 'percentage': 0, 'status': 'No data'}
        
        score = 0
        
        # ID Switches component (30%)
        avg_players = self.metrics['track_consistency']['avg_players_per_frame']
        id_switches_rate = self.metrics['id_switches']['avg_per_frame']
        if avg_players > 0:
            normalized_switch_rate = id_switches_rate / avg_players
        else:
            normalized_switch_rate = id_switches_rate
        id_score = max(0, 1 - normalized_switch_rate * 5)
        score += id_score * 0.3
        
        # Fragmentation component (30%)
        frag_score = 1 - self.metrics['fragmentation']['fragmentation_score']
        score += frag_score * 0.3
        
        # Speed plausibility component (20%)
        speed_score = 1 - min(1.0, self.metrics['speed_plausibility']['implausibility_rate'] * 5)
        score += speed_score * 0.2
        
        # Consistency component (20%)
        consistency_score = self.metrics['track_consistency']['consistency_score']
        score += consistency_score * 0.2
        
        percentage = score * 100
        
        if percentage >= 85:
            rating = 'A (Excellent)'
            status = '✓ Excellent tracking quality'
        elif percentage >= 70:
            rating = 'B (Good)'
            status = '◐ Good tracking, minor improvements needed'
        elif percentage >= 55:
            rating = 'C (Fair)'
            status = '◑ Fair tracking, improvements recommended'
        else:
            rating = 'F (Poor)'
            status = '✗ Poor tracking quality, major issues'
        
        return {
            'rating': rating,
            'percentage': percentage,
            'status': status,
            'components': {
                'id_switches': id_score * 100,
                'fragmentation': frag_score * 100,
                'speed_plausibility': speed_score * 100,
                'consistency': consistency_score * 100
            }
        }
    
    def export_metrics_json(self, output_path: str):
        """Export metrics to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.info(f"✓ Metrics exported to {output_path}")


def evaluate_test_clip(video_path: str, 
                       tracks: Dict,
                       output_json: str = None) -> Dict:
    """
    Convenience function to evaluate a test clip
    
    Args:
        video_path: Path to video file
        tracks: Tracking data
        output_json: Optional path to save JSON report
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = TrackingEvaluator(fps=24)
    metrics = evaluator.evaluate_tracks(tracks)
    evaluator.print_report(verbose=True)
    
    if output_json:
        evaluator.export_metrics_json(output_json)
    
    return metrics

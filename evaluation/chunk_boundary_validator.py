"""
Chunk Boundary ID Continuity Validator

Validates that player IDs are properly maintained across chunk boundaries.
Regression test to ensure the _link_chunk_boundary_ids() logic works correctly.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkBoundaryValidator:
    """Validate ID continuity across chunk boundaries"""
    
    def __init__(self):
        self.metrics = {}
    
    def validate_merged_tracks(self, merged_tracks: Dict) -> Dict:
        """
        Validate that merged tracks show good ID continuity across boundaries
        
        Args:
            merged_tracks: Dictionary with merged player tracks from all chunks
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating chunk boundary ID continuity...")
        
        if 'players' not in merged_tracks or not merged_tracks['players']:
            return {
                'status': 'SKIP',
                'reason': 'No player tracks available',
                'boundary_continuity_score': 0
            }
        
        player_tracks = merged_tracks['players']
        
        # Detect potential chunk boundaries by looking for sudden ID changes
        id_sequences = self._extract_id_sequences(player_tracks)
        boundaries = self._detect_chunk_boundaries(player_tracks)
        
        if not boundaries:
            return {
                'status': 'PASS',
                'reason': 'Single chunk or no boundaries detected',
                'num_boundaries': 0,
                'boundary_continuity_score': 1.0,
                'metrics': {}
            }
        
        # Analyze continuity at each boundary
        boundary_metrics = self._analyze_boundary_continuity(
            player_tracks, boundaries, id_sequences
        )
        
        # Calculate overall continuity score
        continuity_score = self._calculate_continuity_score(boundary_metrics)
        
        self.metrics = {
            'status': 'PASS' if continuity_score >= 0.6 else 'WARN',
            'num_boundaries': len(boundaries),
            'boundary_frames': boundaries,
            'continuity_score': continuity_score,
            'boundary_metrics': boundary_metrics,
            'id_retention_rate': self._calculate_id_retention(id_sequences),
            'recommendation': self._get_recommendation(continuity_score, boundary_metrics)
        }
        
        return self.metrics
    
    def _extract_id_sequences(self, player_tracks: List[Dict]) -> List[Tuple[int, int]]:
        """Extract continuous ID sequences as (start_frame, duration) tuples"""
        id_runs = defaultdict(list)
        
        for frame_num, frame_tracks in enumerate(player_tracks):
            for player_id in frame_tracks.keys():
                id_runs[player_id].append(frame_num)
        
        sequences = []
        for player_id, frames in id_runs.items():
            frames.sort()
            # Group into continuous segments
            segments = []
            start = frames[0]
            prev = frames[0]
            
            for f in frames[1:]:
                if f - prev > 1:
                    segments.append((start, prev - start + 1))
                    start = f
                prev = f
            segments.append((start, prev - start + 1))
            sequences.extend(segments)
        
        return sequences
    
    def _detect_chunk_boundaries(self, player_tracks: List[Dict], 
                                  min_reset_count: int = 5) -> List[int]:
        """
        Detect chunk boundaries by looking for large ID resets
        (sudden loss of many IDs and introduction of new ones)
        """
        boundaries = []
        
        if len(player_tracks) < 2:
            return boundaries
        
        for frame_num in range(1, len(player_tracks)):
            prev_ids = set(player_tracks[frame_num - 1].keys())
            curr_ids = set(player_tracks[frame_num].keys())
            
            # Count disappearances and new arrivals
            disappeared = len(prev_ids - curr_ids)
            new_arrivals = len(curr_ids - prev_ids)
            
            # High simultaneous churn suggests chunk boundary
            if disappeared >= min_reset_count and new_arrivals >= min_reset_count:
                boundaries.append(frame_num)
        
        return boundaries
    
    def _analyze_boundary_continuity(self, 
                                     player_tracks: List[Dict],
                                     boundaries: List[int],
                                     id_sequences: List[Tuple[int, int]]) -> List[Dict]:
        """Analyze ID matching quality at each detected boundary"""
        boundary_metrics = []
        
        for boundary_frame in boundaries:
            if boundary_frame == 0 or boundary_frame >= len(player_tracks):
                continue
            
            prev_frame = player_tracks[boundary_frame - 1]
            curr_frame = player_tracks[boundary_frame]
            
            if not prev_frame or not curr_frame:
                continue
            
            # Calculate preserved ID count (IDs that continue across boundary)
            prev_ids = set(prev_frame.keys())
            curr_ids = set(curr_frame.keys())
            preserved = len(prev_ids & curr_ids)
            total_possible = max(len(prev_ids), len(curr_ids))
            preservation_rate = preserved / total_possible if total_possible > 0 else 0
            
            # Check for position continuity at boundary
            position_jumps = []
            for player_id in (prev_ids & curr_ids):
                prev_pos = prev_frame[player_id].get('position')
                curr_pos = curr_frame[player_id].get('position')
                
                if prev_pos and curr_pos:
                    jump = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                  (curr_pos[1] - prev_pos[1])**2)
                    position_jumps.append(jump)
            
            avg_jump = np.mean(position_jumps) if position_jumps else 0
            max_jump = np.max(position_jumps) if position_jumps else 0
            
            boundary_metrics.append({
                'boundary_frame': boundary_frame,
                'preserved_ids': preserved,
                'total_prev_ids': len(prev_ids),
                'total_curr_ids': len(curr_ids),
                'preservation_rate': preservation_rate,
                'avg_position_jump': float(avg_jump),
                'max_position_jump': float(max_jump),
                'quality': 'GOOD' if preservation_rate > 0.6 else 'FAIR' if preservation_rate > 0.3 else 'POOR'
            })
        
        return boundary_metrics
    
    def _calculate_continuity_score(self, boundary_metrics: List[Dict]) -> float:
        """Calculate overall boundary continuity score (0-1)"""
        if not boundary_metrics:
            return 1.0
        
        preservation_rates = [m['preservation_rate'] for m in boundary_metrics]
        avg_preservation = np.mean(preservation_rates)
        
        # Score based on average preservation rate and consistency
        consistency = 1 - (np.std(preservation_rates) / (np.mean(preservation_rates) + 0.1))
        score = 0.7 * avg_preservation + 0.3 * consistency
        
        return float(np.clip(score, 0, 1))
    
    def _calculate_id_retention(self, id_sequences: List[Tuple[int, int]]) -> float:
        """Calculate how many frames have continuous ID tracking vs short bursts"""
        if not id_sequences:
            return 0.0
        
        durations = [duration for _, duration in id_sequences]
        long_track_threshold = 120  # 5 seconds at 24fps
        long_frames = sum(d for d in durations if d >= long_track_threshold)
        total_frames = sum(durations)
        
        return long_frames / max(total_frames, 1)
    
    def _get_recommendation(self, continuity_score: float, 
                          boundary_metrics: List[Dict]) -> str:
        """Generate actionable recommendation based on metrics"""
        if continuity_score >= 0.85:
            return "✓ Excellent chunk boundary continuity. ID linking is working well."
        elif continuity_score >= 0.65:
            return "◐ Good continuity. Minor ID losses at boundaries, acceptable for most use cases."
        elif continuity_score >= 0.40:
            return "◑ Fair continuity. Consider tuning _link_chunk_boundary_ids() max_dist parameter."
        else:
            return "✗ Poor continuity. Chunk boundary linking may be ineffective; review camera movement or increase max_dist."
    
    def print_report(self) -> str:
        """Generate human-readable validation report"""
        if not self.metrics:
            return "No validation metrics available. Run validate_merged_tracks first."
        
        report = []
        report.append("\n" + "=" * 70)
        report.append("CHUNK BOUNDARY ID CONTINUITY VALIDATION REPORT")
        report.append("=" * 70)
        
        report.append(f"\nStatus: {self.metrics['status']}")
        report.append(f"Number of Boundaries Detected: {self.metrics.get('num_boundaries', 0)}")
        report.append(f"Overall Continuity Score: {self.metrics.get('continuity_score', 0):.3f}")
        report.append(f"ID Retention Rate (long tracks): {self.metrics.get('id_retention_rate', 0):.3f}")
        
        if self.metrics.get('boundary_metrics'):
            report.append("\n[BOUNDARY ANALYSIS]")
            for bm in self.metrics['boundary_metrics']:
                report.append(f"\n  Frame {bm['boundary_frame']}:")
                report.append(f"    Preserved IDs: {bm['preserved_ids']}/{bm['total_prev_ids']} → {bm['total_curr_ids']}")
                report.append(f"    Preservation Rate: {bm['preservation_rate']:.2%}")
                report.append(f"    Avg Position Jump: {bm['avg_position_jump']:.1f}px")
                report.append(f"    Max Position Jump: {bm['max_position_jump']:.1f}px")
                report.append(f"    Quality: {bm['quality']}")
        
        report.append(f"\n[RECOMMENDATION]")
        report.append(f"  {self.metrics.get('recommendation', 'No recommendation')}")
        
        report.append("\n" + "=" * 70)
        
        report_str = "\n".join(report)
        logger.info(report_str)
        return report_str
    
    def export_json(self, output_path: str):
        """Export validation metrics to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.info(f"✓ Validation report exported to {output_path}")


def validate_chunk_boundary_continuity(merged_tracks: Dict, 
                                       output_json: str = None) -> Dict:
    """
    Convenience function to validate chunk boundary ID continuity
    
    Args:
        merged_tracks: Merged track dictionary from pipeline
        output_json: Optional path to save JSON report
        
    Returns:
        Dictionary with validation metrics
    """
    validator = ChunkBoundaryValidator()
    metrics = validator.validate_merged_tracks(merged_tracks)
    validator.print_report()
    
    if output_json:
        validator.export_json(output_json)
    
    return metrics

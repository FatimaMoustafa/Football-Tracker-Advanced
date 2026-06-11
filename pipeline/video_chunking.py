"""
Video Chunking Pipeline using OpenCV
Splits large videos into 60s chunks, processes each independently, merges results
"""

import cv2
import os
import json
import shutil
import math
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoChunker:
    """Handles video splitting with OpenCV"""

    def __init__(self, chunk_duration: int = 60):
        """
        Initialize video chunker

        Args:
            chunk_duration: Duration of each chunk in seconds (default 60s)
        """
        self.chunk_duration = chunk_duration

    def split_video(self, input_path: str, output_dir: str) -> List[str]:
        """
        Split video into chunks using OpenCV.

        Reads the source video frame-by-frame and writes each chunk to a
        separate .mp4 file using the same codec/resolution as the source.

        Args:
            input_path: Path to input video
            output_dir: Directory to save chunks

        Returns:
            List of chunk file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if fps <= 0:
            cap.release()
            raise RuntimeError(f"Cannot determine FPS for {input_path}")

        duration = frame_count / fps
        frames_per_chunk = int(fps * self.chunk_duration)
        num_chunks = math.ceil(frame_count / frames_per_chunk)

        logger.info(f"Video duration: {duration:.2f}s  |  FPS: {fps:.2f}  |  "
                    f"Resolution: {width}x{height}")
        logger.info(f"Splitting into {num_chunks} chunks of {self.chunk_duration}s "
                    f"({frames_per_chunk} frames each)")

        chunk_paths = []
        writer: cv2.VideoWriter | None = None
        chunk_idx = -1
        current_chunk_frames = 0

        for frame_num in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Start a new chunk when needed
            if frame_num % frames_per_chunk == 0:
                # Close the previous writer
                if writer is not None:
                    writer.release()
                    logger.info(f"✓ Chunk {chunk_idx + 1} created successfully")

                chunk_idx += 1
                chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.mp4")
                writer = cv2.VideoWriter(chunk_path, fourcc, fps, (width, height))

                if not writer.isOpened():
                    cap.release()
                    raise RuntimeError(f"Cannot create chunk file: {chunk_path}")

                chunk_paths.append(chunk_path)
                current_chunk_frames = 0
                logger.info(f"Creating chunk {chunk_idx + 1}/{num_chunks}: {chunk_path}")

            writer.write(frame)
            current_chunk_frames += 1

        # Release the last writer
        if writer is not None:
            writer.release()
            logger.info(f"✓ Chunk {chunk_idx + 1} created successfully")

        cap.release()
        logger.info(f"✓ All {len(chunk_paths)} chunks created successfully")
        return chunk_paths

    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps > 0:
            return frame_count / fps

        raise RuntimeError(f"Cannot determine video duration for {video_path}")


class JSONMerger:
    """Merge JSON stats from multiple chunks with frame offset correction"""
    
    @staticmethod
    def merge_chunk_stats(chunk_stats_list: List[Dict], 
                         chunk_duration: int = 60,
                         fps: int = 24) -> Dict:
        """
        Merge tracking stats from multiple chunks
        
        Args:
            chunk_stats_list: List of stats dicts from each chunk
            chunk_duration: Duration of each chunk in seconds
            fps: Frames per second of video
            
        Returns:
            Merged stats dictionary
        """
        if not chunk_stats_list:
            return {}
        
        merged_stats = {
            'total_frames': 0,
            'total_players': set(),
            'total_id_switches': 0,
            'chunks_processed': len(chunk_stats_list),
            'id_switches_per_chunk': [],
            'fragmentation_per_chunk': [],
            'avg_track_length': 0,
            'chunk_transitions': []
        }
        
        frame_offset = 0
        prev_chunk_player_ids = {}
        
        for chunk_idx, stats in enumerate(chunk_stats_list):
            if not stats:
                continue
            
            # Track total frames
            chunk_frames = stats.get('total_frames', 0)
            merged_stats['total_frames'] += chunk_frames
            
            # Track player IDs
            player_ids = set(stats.get('player_ids', []))
            merged_stats['total_players'].update(player_ids)
            
            # Accumulate ID switches
            id_switches = stats.get('id_switches', 0)
            merged_stats['total_id_switches'] += id_switches
            merged_stats['id_switches_per_chunk'].append({
                'chunk': chunk_idx,
                'switches': id_switches
            })
            
            # Track fragmentation
            fragmentation = stats.get('fragmentation_score', 0)
            merged_stats['fragmentation_per_chunk'].append({
                'chunk': chunk_idx,
                'fragmentation': fragmentation
            })
            
            # Detect ID remapping at chunk transitions
            if chunk_idx > 0 and prev_chunk_player_ids:
                transition_info = {
                    'between_chunks': f"{chunk_idx-1}_to_{chunk_idx}",
                    'prev_players': list(prev_chunk_player_ids.keys()),
                    'curr_players': list(player_ids),
                    'players_lost': list(prev_chunk_player_ids.keys() - player_ids),
                    'players_gained': list(player_ids - prev_chunk_player_ids.keys())
                }
                merged_stats['chunk_transitions'].append(transition_info)
            
            prev_chunk_player_ids = {pid: frame_offset for pid in player_ids}
            frame_offset += chunk_frames
        
        merged_stats['total_players'] = list(merged_stats['total_players'])
        merged_stats['total_unique_players'] = len(merged_stats['total_players'])
        
        if merged_stats['total_frames'] > 0:
            merged_stats['avg_id_switches_per_frame'] = (
                merged_stats['total_id_switches'] / merged_stats['total_frames']
            )
        
        logger.info(f"Merged {len(chunk_stats_list)} chunks: "
                   f"{merged_stats['total_frames']} total frames, "
                   f"{merged_stats['total_id_switches']} total ID switches")
        
        return merged_stats
    
    @staticmethod
    def save_merged_stats(stats: Dict, output_path: str):
        """Save merged stats to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"✓ Merged stats saved to {output_path}")


class ChunkProcessor:
    """Process individual video chunks through the tracking pipeline"""
    
    def __init__(self, tracker_instance):
        """
        Initialize chunk processor
        
        Args:
            tracker_instance: Tracker object for processing frames
        """
        self.tracker = tracker_instance
    
    def process_chunk(self, frames: List) -> Dict:
        """
        Process a single chunk and return tracking stats
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary containing tracking statistics
        """
        if not frames:
            return {}
        
        # Track detection stats
        stats = {
            'total_frames': len(frames),
            'player_ids': set(),
            'id_switches': 0,
            'fragmentation_score': 0,
            'tracks_per_frame': []
        }
        
        # Get tracks for this chunk
        try:
            tracks = self.tracker.get_object_tracks(frames, read_from_stub=False)
            
            # Analyze tracking quality
            for frame_num, frame_tracks in enumerate(tracks.get('players', [])):
                frame_player_ids = set(frame_tracks.keys())
                stats['player_ids'].update(frame_player_ids)
                stats['tracks_per_frame'].append(len(frame_player_ids))
            
            # Calculate fragmentation
            if stats['tracks_per_frame']:
                avg_players = sum(stats['tracks_per_frame']) / len(stats['tracks_per_frame'])
                fragmentation = len(set(stats['player_ids'])) / (avg_players + 1e-6)
                stats['fragmentation_score'] = min(1.0, fragmentation)
            
            stats['player_ids'] = list(stats['player_ids'])
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
        
        return stats


def chunk_and_process_video(video_path: str, 
                           output_dir: str = 'output_videos',
                           tracker_instance=None) -> Tuple[str, Dict]:
    """
    Complete pipeline: chunk video → process → merge stats
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for results
        tracker_instance: Tracker instance for processing
        
    Returns:
        Tuple of (output_video_path, merged_stats)
    """
    logger.info("=" * 60)
    logger.info("VIDEO CHUNKING PIPELINE")
    logger.info("=" * 60)
    
    # Create directories
    chunks_dir = os.path.join(output_dir, 'chunks')
    stats_dir = os.path.join(output_dir, 'stats')
    Path(chunks_dir).mkdir(parents=True, exist_ok=True)
    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Split video
        logger.info("\n[STEP 1] Splitting video into 60s chunks...")
        chunker = VideoChunker(chunk_duration=60)
        chunk_paths = chunker.split_video(video_path, chunks_dir)
        
        # Step 2: Process chunks
        logger.info(f"\n[STEP 2] Processing {len(chunk_paths)} chunks...")
        chunk_stats_list = []
        processor = ChunkProcessor(tracker_instance) if tracker_instance else None
        
        for i, chunk_path in enumerate(chunk_paths):
            logger.info(f"\nProcessing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")
            
            if processor:
                frames = []
                cap = cv2.VideoCapture(chunk_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                
                stats = processor.process_chunk(frames)
                chunk_stats_list.append(stats)
                
                # Save chunk stats
                chunk_stats_path = os.path.join(stats_dir, f"chunk_{i:04d}_stats.json")
                with open(chunk_stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                logger.info(f"✓ Chunk {i+1} processed: {len(frames)} frames, "
                           f"{len(stats.get('player_ids', []))} unique players")
        
        # Step 3: Merge stats
        logger.info(f"\n[STEP 3] Merging stats from {len(chunk_stats_list)} chunks...")
        merged_stats = JSONMerger.merge_chunk_stats(chunk_stats_list)
        
        # Save merged stats
        merged_stats_path = os.path.join(stats_dir, 'merged_stats.json')
        JSONMerger.save_merged_stats(merged_stats, merged_stats_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total frames processed: {merged_stats['total_frames']}")
        logger.info(f"Total unique players: {merged_stats['total_unique_players']}")
        logger.info(f"Total ID switches: {merged_stats['total_id_switches']}")
        logger.info(f"Avg ID switches/frame: {merged_stats.get('avg_id_switches_per_frame', 0):.4f}")
        logger.info("=" * 60 + "\n")
        
        return merged_stats_path, merged_stats
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        # Cleanup chunks
        if os.path.exists(chunks_dir):
            logger.info(f"Cleaning up chunk files...")
            shutil.rmtree(chunks_dir)
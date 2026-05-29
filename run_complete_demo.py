"""
Complete Demo and Test Script
Shows how to use all components: chunking, evaluation, and homography picker
"""

import os
import json
import argparse
import cv2
from pathlib import Path
import logging
from typing import Optional

from run_chunked_pipeline import ChunkedPipeline
from run_evaluation import EvaluationHarness
from tools.homography_picker import interactive_homography_picker, HomographyPicker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompleteTrackingWorkflow:
    """Complete workflow combining all tools"""
    
    def __init__(self, model_path: str = 'models/best.pt'):
        # Convert relative paths to absolute based on script location
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, model_path)
        
        self.model_path = model_path
        self.pipeline = ChunkedPipeline(model_path=model_path)
        self.evaluator = EvaluationHarness(model_path=model_path)
    
    def print_banner(self, title: str):
        """Print formatted banner"""
        print("\n" + "=" * 70)
        print(title.center(70))
        print("=" * 70)
    
    def run_complete_workflow(self,
                             video_path: str,
                             output_dir: str = 'workflow_output',
                             homography_file: Optional[str] = None,
                             setup_homography: bool = False) -> dict:
        """
        Run complete workflow: optional homography setup, chunked processing, evaluation
        
        Args:
            video_path: Input video path
            output_dir: Output directory
            homography_file: Path to existing homography matrix
            setup_homography: Whether to interactively setup homography
            
        Returns:
            Dictionary with workflow results
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results = {}
        
        # Step 1: Homography (optional)
        homography_file = os.path.join(output_dir, 'homography_matrix.json')
        if setup_homography and not os.path.exists(homography_file):
            self.print_banner("STEP 1: HOMOGRAPHY CALIBRATION")
            
            logger.info("Extracting first frame from video...")
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_path = os.path.join(output_dir, 'reference_frame.jpg')
                cv2.imwrite(frame_path, frame)
                
                logger.info(f"\nInteractive homography picker will now open...")
                logger.info(f"Click 4 pitch corners in order: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
                logger.info(f"Right-click to undo, SPACE to confirm\n")
                
                homography_data = interactive_homography_picker(
                    image_path=frame_path,
                    output_json=homography_file,
                    court_width=68,
                    court_length=23.32
                )
                
                if homography_data:
                    results['homography'] = {
                        'status': 'success',
                        'file': homography_file,
                        'matrix_shape': str(homography_data['homography_matrix'])[:50] + '...'
                    }
                    logger.info("✓ Homography setup completed")
                else:
                    logger.warning("Homography setup skipped")
                    results['homography'] = {'status': 'skipped'}
            else:
                logger.error("Could not extract frame from video")
                results['homography'] = {'status': 'failed'}
        else:
            logger.info("✓ Using existing homography matrix")
            results['homography'] = {
                'status': 'existing',
                'file': homography_file
            }
        
        # Step 2: Chunked Processing
        self.print_banner("STEP 2: CHUNKED VIDEO PROCESSING")
        
        logger.info(f"Processing video with memory-efficient chunking...")
        logger.info(f"Input: {video_path}")
        
        merged_tracks = None
        try:
            processing_dir = os.path.join(output_dir, 'processing')
            stats = self.pipeline.process_video_chunked(
                video_path=video_path,
                output_dir=processing_dir,
                use_stub=False,
                return_merged_tracks=True
            )
            merged_tracks = stats.pop('tracks', None)
            
            results['processing'] = {
                'status': 'success',
                'output_dir': processing_dir,
                'chunks_processed': stats.get('total_chunks', 'N/A'),
                'score': stats.get('combined_score', 'N/A'),
                'rating': stats.get('rating', 'N/A')
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            results['processing'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Step 3: Evaluation
        self.print_banner("STEP 3: TRACKING QUALITY EVALUATION")
        
        logger.info("Running tracking evaluation harness...")
        
        try:
            eval_dir = os.path.join(output_dir, 'evaluation')
            metrics = self.evaluator.evaluate_video(
                video_path=video_path,
                output_dir=eval_dir,
                existing_tracks=merged_tracks
            )
            
            results['evaluation'] = {
                'status': 'success',
                'output_dir': eval_dir,
                'total_frames': metrics['summary']['total_frames'],
                'unique_players': metrics['summary']['unique_players'],
                'id_switches': metrics['id_switches']['total_switches'],
                'fragmentation_score': metrics['fragmentation']['fragmentation_score'],
                'implausible_movements': metrics['speed_plausibility']['implausible_movements']
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            results['evaluation'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Save workflow results
        results_file = os.path.join(output_dir, 'workflow_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def print_results_summary(self, results: dict):
        """Print formatted results summary"""
        self.print_banner("WORKFLOW RESULTS SUMMARY")
        
        print("\n Homography Calibration:")
        hom = results.get('homography', {})
        if hom.get('status') == 'success':
            print(f"  ✓ Completed - {hom.get('file')}")
        else:
            print(f"   {hom.get('status', 'unknown')}")
        
        print("\n Chunked Processing:")
        proc = results.get('processing', {})
        if proc.get('status') == 'success':
            print(f"  ✓ Completed")
            print(f"    Chunks: {proc.get('chunks_processed')}")
            print(f"    Score: {proc.get('score', 'N/A')}")
            print(f"    Rating: {proc.get('rating', 'N/A')}")
            print(f"    Output: {proc.get('output_dir')}")
        else:
            print(f"  ✗ {proc.get('status', 'unknown')}: {proc.get('error', 'No error details')}")
        
        print("\n Evaluation:")
        eval_res = results.get('evaluation', {})
        if eval_res.get('status') == 'success':
            print(f"  ✓ Completed")
            print(f"    Total frames: {eval_res.get('total_frames')}")
            print(f"    Unique players: {eval_res.get('unique_players')}")
            print(f"    ID switches: {eval_res.get('id_switches')}")
            print(f"    Fragmentation: {eval_res.get('fragmentation_score', 'N/A'):.3f}")
            print(f"    Implausible moves: {eval_res.get('implausible_movements')}")
            print(f"    Output: {eval_res.get('output_dir')}")
        else:
            print(f"  ✗ {eval_res.get('status', 'unknown')}: {eval_res.get('error', 'No error details')}")
        
        print("\n" + "=" * 70 + "\n")


def test_quick_evaluation(video_path: str, output_dir: str = 'quick_eval'):
    """Quick evaluation only (no chunking)"""
    logger.info("Running quick evaluation...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluator = EvaluationHarness()
    metrics = evaluator.evaluate_video(video_path, output_dir=output_dir)
    
    print("\nQuick Evaluation Results:")
    print(json.dumps({
        'frames': metrics['summary']['total_frames'],
        'players': metrics['summary']['unique_players'],
        'id_switches': metrics['id_switches']['total_switches'],
    }, indent=2))


def test_chunking_only(video_path: str, output_dir: str = 'chunking_test'):
    """Test chunking pipeline only"""
    logger.info("Running chunking test...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pipeline = ChunkedPipeline()
    stats = pipeline.process_video_chunked(video_path, output_dir=output_dir)
    
    print("\nChunking Test Results:")
    print(json.dumps(stats, indent=2, default=str))


def test_homography_only(image_path: str, output_dir: str = 'homography_test'):
    """Test homography picker only"""
    logger.info("Running homography picker test...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'homography_matrix.json')
    
    homography_data = interactive_homography_picker(
        image_path=image_path,
        output_json=output_file
    )
    
    if homography_data:
        print(f"\n✓ Homography test successful - saved to {output_file}")
    else:
        print("\n✗ Homography test cancelled")


def main():
    parser = argparse.ArgumentParser(
        description='Complete Football Tracking Workflow'
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--full', action='store_true',
                           help='Run complete workflow (homography + chunking + evaluation)')
    mode_group.add_argument('--eval-only', action='store_true',
                           help='Run quick evaluation only (fast)')
    mode_group.add_argument('--chunk-only', action='store_true',
                           help='Run chunking pipeline only')
    mode_group.add_argument('--homography-only', action='store_true',
                           help='Run homography picker only')
    
    # Arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input video or image file')
    parser.add_argument('--output', '-o', default='demo_output',
                       help='Output directory')
    parser.add_argument('--model', default='models/best.pt',
                       help='Model path')
    parser.add_argument('--skip-homography', action='store_true',
                       help='Skip homography setup in full mode')
    
    args = parser.parse_args()
    
    logger.info("Football Tracking AI Agent - Complete Workflow Demo")
    logger.info(f"Model: {args.model}")
    logger.info(f"Input: {args.input}")
    
    try:
        if args.full:
            workflow = CompleteTrackingWorkflow(model_path=args.model)
            results = workflow.run_complete_workflow(
                video_path=args.input,
                output_dir=args.output,
                setup_homography=not args.skip_homography
            )
            workflow.print_results_summary(results)
        
        elif args.eval_only:
            test_quick_evaluation(args.input, args.output)
        
        elif args.chunk_only:
            test_chunking_only(args.input, args.output)
        
        elif args.homography_only:
            test_homography_only(args.input, args.output)
    
    except KeyboardInterrupt:
        logger.warning("Workflow interrupted by user")
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

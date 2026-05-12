"""
Homography Picker Tool Script - Interactive corner selection for perspective transform
"""

import os
import argparse
import cv2
from pathlib import Path
import logging

from tools.homography_picker import interactive_homography_picker, HomographyPicker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_first_frame(video_path: str, output_dir: str = 'temp') -> str:
    """Extract first frame from video for picker"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    frame_path = os.path.join(output_dir, 'first_frame.jpg')
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(frame_path, frame)
        logger.info(f"✓ First frame extracted to {frame_path}")
        return frame_path
    else:
        raise ValueError(f"Cannot extract frame from {video_path}")


def main():
    parser = argparse.ArgumentParser(description='Homography Picker Tool')
    parser.add_argument('--input', '-i', required=True, 
                       help='Input image or video file')
    parser.add_argument('--output', '-o', default='config/homography_matrix.json',
                       help='Output JSON file for homography matrix')
    parser.add_argument('--width', type=float, default=68,
                       help='Court width in meters (default: 68)')
    parser.add_argument('--length', type=float, default=23.32,
                       help='Court length in meters (default: 23.32)')
    
    args = parser.parse_args()
    
    # Handle video input
    input_path = args.input
    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        logger.info("Video file detected, extracting first frame...")
        input_path = extract_first_frame(args.input)
    
    # Run homography picker
    homography_data = interactive_homography_picker(
        image_path=input_path,
        output_json=args.output,
        court_width=args.width,
        court_length=args.length
    )
    
    if homography_data:
        logger.info("\n" + "=" * 70)
        logger.info("✓ HOMOGRAPHY PICKER SUCCESSFUL")
        logger.info("=" * 70)
        logger.info(f"Transform matrix saved to: {args.output}")
        logger.info("\nYou can now use this matrix in your pipeline with:")
        logger.info(f"  loader = HomographyPicker()")
        logger.info(f"  data = loader.load_from_file('{args.output}')")
        logger.info("=" * 70 + "\n")
    else:
        logger.error("Homography picker cancelled")


if __name__ == '__main__':
    main()

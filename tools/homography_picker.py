"""
Interactive Homography Picker Tool
Click 4 pitch corners to define perspective transform matrix
Saves transform to JSON for reuse
"""

import cv2
import json
import numpy as np
from pathlib import Path
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HomographyPicker:
    """Interactive tool to select pitch corners and compute homography"""
    
    def __init__(self, court_width: float = 68, court_length: float = 23.32):
        """
        Initialize homography picker
        
        Args:
            court_width: Width of court in meters (default 68m for football)
            court_length: Length of court in meters (default 23.32m for football)
        """
        self.court_width = court_width
        self.court_length = court_length
        
        self.points = []
        self.image = None
        self.original_image = None
        self.window_name = "Homography Picker - Click 4 Corners (TL, TR, BR, BL)"
        self.homography_matrix = None
        self.target_vertices = None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for clicking points"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
                
                # Draw circle at clicked point
                cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
                
                # Draw point number
                cv2.putText(self.image, f"{len(self.points)}", (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw line connecting points
                if len(self.points) > 1:
                    cv2.line(self.image, tuple(self.points[-2]), (x, y), (0, 255, 0), 2)
                
                # Auto-close polygon on 4th point
                if len(self.points) == 4:
                    cv2.line(self.image, tuple(self.points[3]), tuple(self.points[0]), (0, 255, 0), 2)
                    logger.info("✓ 4 points selected. Computing homography...")
                
                cv2.imshow(self.window_name, self.image)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self.image = self.original_image.copy()
                
                for i, point in enumerate(self.points):
                    cv2.circle(self.image, tuple(point), 5, (0, 255, 0), -1)
                    cv2.putText(self.image, f"{i+1}", (point[0] + 10, point[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(self.points) > 1:
                    for i in range(len(self.points) - 1):
                        cv2.line(self.image, tuple(self.points[i]), tuple(self.points[i+1]), 
                                (0, 255, 0), 2)
                
                logger.info(f"Point removed. {len(self.points)} points remaining.")
                cv2.imshow(self.window_name, self.image)
    
    def pick_points_from_image(self, image_path: str) -> np.ndarray:
        """
        Interactively pick 4 corner points from image
        
        Args:
            image_path: Path to image/video frame
            
        Returns:
            Array of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        logger.info(f"Loading image: {image_path}")
        
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize for display if too large
        height, width = self.original_image.shape[:2]
        if width > 1280 or height > 720:
            scale = min(1280 / width, 720 / height)
            self.original_image = cv2.resize(self.original_image, 
                                             (int(width * scale), int(height * scale)))
            logger.info(f"Image resized to {self.original_image.shape[:2]}")
        
        self.image = self.original_image.copy()
        self.points = []
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Display instructions
        cv2.putText(self.image, "Click 4 pitch corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(self.image, "Right-click to undo. Press SPACE when done.",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow(self.window_name, self.image)
        
        logger.info("Instructions:")
        logger.info("  - LEFT click to add points")
        logger.info("  - RIGHT click to undo")
        logger.info("  - SPACE to complete when 4 points are selected")
        
        # Wait for user to complete selection
        while len(self.points) < 4:
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC
                logger.warning("Cancelled by user")
                cv2.destroyAllWindows()
                return None
        
        # Wait for SPACE confirmation
        logger.info("Waiting for confirmation (SPACE)...")
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == 32:  # SPACE
                break
            elif key == 27:  # ESC
                logger.warning("Cancelled by user")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        points_array = np.array(self.points, dtype=np.float32)
        logger.info(f"✓ Points selected: {points_array.tolist()}")
        
        return points_array
    
    def compute_homography(self, pixel_vertices: np.ndarray) -> Dict:
        """
        Compute perspective transform matrix
        
        Args:
            pixel_vertices: 4x2 array of pixel coordinates
            
        Returns:
            Dictionary with transform matrix and metadata
        """
        if pixel_vertices.shape != (4, 2):
            raise ValueError("Must provide exactly 4 points")
        
        # Define target court coordinates
        self.target_vertices = np.array([
            [0, self.court_width],
            [0, 0],
            [self.court_length, 0],
            [self.court_length, self.court_width]
        ], dtype=np.float32)
        
        # Compute perspective transform
        self.homography_matrix = cv2.getPerspectiveTransform(
            pixel_vertices.astype(np.float32),
            self.target_vertices.astype(np.float32)
        )
        
        logger.info("✓ Homography matrix computed")
        logger.info(f"\nHomography Matrix ({self.homography_matrix.shape}):")
        logger.info(self.homography_matrix)
        
        return {
            'homography_matrix': self.homography_matrix.tolist(),
            'pixel_vertices': pixel_vertices.tolist(),
            'target_vertices': self.target_vertices.tolist(),
            'court_width_m': self.court_width,
            'court_length_m': self.court_length,
            'description': 'Perspective transform from pixel to court coordinates'
        }
    
    def verify_transform(self, test_image_path: str = None) -> np.ndarray:
        """
        Verify transform by visualizing transformed perspective
        
        Args:
            test_image_path: Optional test image to verify on
            
        Returns:
            Transformed image
        """
        if self.homography_matrix is None:
            raise ValueError("Homography not computed yet")
        
        if test_image_path is None:
            logger.warning("No test image provided, skipping verification")
            return None
        
        image = cv2.imread(test_image_path)
        if image is None:
            logger.warning(f"Cannot read test image: {test_image_path}")
            return None
        
        # Apply perspective transform
        transformed = cv2.warpPerspective(image, self.homography_matrix, 
                                         (int(self.court_length * 10), int(self.court_width * 10)))
        
        logger.info("✓ Transform verification complete")
        
        return transformed
    
    def save_to_file(self, output_path: str, homography_data: Dict):
        """
        Save homography matrix to JSON file
        
        Args:
            output_path: Path to save JSON file
            homography_data: Dictionary with homography data
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(homography_data, f, indent=2)
        
        logger.info(f"✓ Homography matrix saved to {output_path}")
    
    def load_from_file(self, json_path: str) -> Dict:
        """
        Load homography matrix from JSON file
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Dictionary with homography data
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.homography_matrix = np.array(data['homography_matrix'], dtype=np.float32)
        self.target_vertices = np.array(data['target_vertices'], dtype=np.float32)
        
        logger.info(f"✓ Homography matrix loaded from {json_path}")
        
        return data


def interactive_homography_picker(image_path: str, 
                                 output_json: str = 'config/homography_matrix.json',
                                 court_width: float = 68,
                                 court_length: float = 23.32) -> Dict:
    """
    Complete interactive homography picking workflow
    
    Args:
        image_path: Path to first frame or any frame from video
        output_json: Path to save homography matrix
        court_width: Court width in meters
        court_length: Court length in meters
        
    Returns:
        Dictionary with homography data
    """
    logger.info("=" * 70)
    logger.info("HOMOGRAPHY PICKER TOOL")
    logger.info("=" * 70)
    
    picker = HomographyPicker(court_width=court_width, court_length=court_length)
    
    # Step 1: Pick points
    logger.info("\n[STEP 1] Selecting corner points...")
    pixel_vertices = picker.pick_points_from_image(image_path)
    
    if pixel_vertices is None:
        logger.error("Point selection cancelled")
        return None
    
    # Step 2: Compute homography
    logger.info("\n[STEP 2] Computing homography matrix...")
    homography_data = picker.compute_homography(pixel_vertices)
    
    # Step 3: Save to file
    logger.info("\n[STEP 3] Saving homography matrix...")
    picker.save_to_file(output_json, homography_data)
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ HOMOGRAPHY PICKER COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Saved to: {output_json}\n")
    
    return homography_data

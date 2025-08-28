"""
Face Detection Module for NAVADA
This module provides face detection capabilities using OpenCV's Haar Cascades.
It can detect faces, eyes, and smiles in images and return detailed statistics.
"""

import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy for numerical operations on arrays
from typing import Tuple, List, Dict, Optional, Union  # Type hints for better code documentation
import os  # Operating system interface for file path operations
import logging  # Logging module for error tracking

# Configure logging for this module
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    A class to handle face detection using OpenCV's Haar Cascade classifiers.
    This detector can identify faces, eyes, and smiles in images.
    """
    
    def __init__(self):
        """
        Initialize the FaceDetector with pre-trained Haar Cascade classifiers.
        Loads classifiers for face, eye, and smile detection.
        """
        try:
            # Load the pre-trained Haar Cascade classifier for frontal face detection
            # This XML file contains trained patterns for detecting frontal faces
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Load the classifier for eye detection
            # This works best when applied to face regions
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
            # Load the classifier for smile detection
            # This detects smiling expressions in face regions
            self.smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml'
            )
            
            # Verify that classifiers loaded successfully
            if self.face_cascade.empty():
                raise ValueError("Failed to load face cascade classifier")
            if self.eye_cascade.empty():
                raise ValueError("Failed to load eye cascade classifier")
            if self.smile_cascade.empty():
                raise ValueError("Failed to load smile cascade classifier")
                
            logger.info("Face detection classifiers loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing face detector: {str(e)}")
            raise
        
    def detect_faces(self, image: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        Detect faces in an image and return an annotated image with statistics.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image as a NumPy array (can be grayscale or color)
            
        Returns:
        --------
        Tuple[np.ndarray, Dict]
            - Annotated image with face detection boxes and labels
            - Dictionary containing detection statistics and face details
        """
        # Input validation - check if image is provided and valid
        if image is None:
            logger.warning("No image provided for face detection")
            return np.zeros((480, 640, 3), dtype=np.uint8), {
                'total_faces': 0,
                'faces': [],
                'detection_method': 'Haar Cascade',
                'features_detected': {'eyes': 0, 'smiles': 0}
            }
            
        # Ensure image is a numpy array
        if not isinstance(image, np.ndarray):
            logger.error("Image must be a numpy array")
            raise TypeError("Image must be a numpy array")
            
        # Check if image is empty
        if image.size == 0:
            logger.warning("Empty image provided")
            return image, {
                'total_faces': 0,
                'faces': [],
                'detection_method': 'Haar Cascade',
                'features_detected': {'eyes': 0, 'smiles': 0}
            }
        
        # Convert grayscale images to RGB for consistent output
        # Check the number of dimensions to determine if image is grayscale
        if len(image.shape) == 2:  # Grayscale image (height, width)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:  # Color image (height, width, channels)
            img_rgb = image.copy()  # Create a copy to avoid modifying original
        else:
            logger.error(f"Invalid image shape: {image.shape}")
            raise ValueError(f"Invalid image shape: {image.shape}")
            
        # Convert to grayscale for detection algorithms
        # Haar Cascades work on grayscale images for better performance
        try:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        except cv2.error as e:
            logger.error(f"Error converting image to grayscale: {str(e)}")
            return img_rgb, {
                'total_faces': 0,
                'faces': [],
                'detection_method': 'Haar Cascade',
                'features_detected': {'eyes': 0, 'smiles': 0}
            }
        
        # Detect faces using the Haar Cascade classifier
        # Parameters control detection sensitivity and performance
        faces = self.face_cascade.detectMultiScale(
            gray,  # Grayscale image to search
            scaleFactor=1.1,  # Image pyramid scaling factor (1.1 = 10% reduction each level)
            minNeighbors=5,  # Minimum neighbors for detection confidence
            minSize=(30, 30)  # Minimum face size in pixels
        )
        
        # List to store detailed information about each detected face
        face_details = []
        
        # Process each detected face
        for idx, (x, y, w, h) in enumerate(faces):
            # Draw a magenta rectangle around the detected face
            # Parameters: image, top-left corner, bottom-right corner, color (BGR), thickness
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 255), 3)
            
            # Add a label above each face
            # Parameters: image, text, position, font, scale, color, thickness
            cv2.putText(
                img_rgb, 
                f"Face {idx+1}",  # Label text
                (x, y-10),  # Position (above the rectangle)
                cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                0.7,  # Font scale
                (255, 0, 255),  # Color (magenta in RGB)
                2  # Thickness
            )
            
            # Extract Region of Interest (ROI) for face area
            # This isolates the face region for feature detection
            roi_gray = gray[y:y+h, x:x+w]  # Grayscale ROI for detection
            roi_color = img_rgb[y:y+h, x:x+w]  # Color ROI for drawing
            
            # Detect eyes within the face region
            # Using different parameters for eye detection (more sensitive)
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,  # Smaller scale factor for finer detection
                minNeighbors=3  # Fewer neighbors required
            )
            eye_count = len(eyes)  # Count the number of detected eyes
            
            # Draw green rectangles around detected eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    roi_color,  # Draw on the color ROI
                    (ex, ey),  # Top-left corner
                    (ex+ew, ey+eh),  # Bottom-right corner
                    (0, 255, 0),  # Green color in RGB
                    2  # Thickness
                )
            
            # Detect smiles within the face region
            # Smile detection requires different parameters
            smiles = self.smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.8,  # Larger scale factor for smile detection
                minNeighbors=20  # More neighbors required for confidence
            )
            has_smile = len(smiles) > 0  # Boolean flag for smile presence
            
            # Draw yellow rectangles around detected smiles
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(
                    roi_color,  # Draw on the color ROI
                    (sx, sy),  # Top-left corner
                    (sx+sw, sy+sh),  # Bottom-right corner
                    (0, 255, 255),  # Yellow color in RGB
                    2  # Thickness
                )
            
            # Store detailed information about this face
            face_details.append({
                'face_id': idx + 1,  # Sequential face ID starting from 1
                'position': {  # Face bounding box coordinates
                    'x': int(x),  # X coordinate of top-left corner
                    'y': int(y),  # Y coordinate of top-left corner
                    'width': int(w),  # Width of face bounding box
                    'height': int(h)  # Height of face bounding box
                },
                'eyes_detected': eye_count,  # Number of eyes detected
                'smile_detected': has_smile,  # Whether a smile was detected
                'confidence': 0.95  # Placeholder confidence score
            })
        
        # Compile comprehensive statistics about all detected faces
        stats = {
            'total_faces': len(faces),  # Total number of faces detected
            'faces': face_details,  # List of detailed face information
            'detection_method': 'Haar Cascade',  # Method used for detection
            'features_detected': {  # Aggregate feature statistics
                'eyes': sum(f['eyes_detected'] for f in face_details),  # Total eyes
                'smiles': sum(1 for f in face_details if f['smile_detected'])  # Total smiles
            }
        }
        
        return img_rgb, stats  # Return annotated image and statistics

    def analyze_demographics(self, face_stats: Optional[Dict]) -> str:
        """
        Create a demographic analysis report based on face detection statistics.
        
        Parameters:
        -----------
        face_stats : Dict
            Dictionary containing face detection statistics
            
        Returns:
        --------
        str
            Formatted text analysis of detected faces and their features
        """
        # Handle case where no statistics are provided
        if not face_stats:
            return "No face detection data available."
            
        # Handle case where no faces were detected
        if face_stats.get('total_faces', 0) == 0:
            return "No faces detected in the image."
        
        # Build analysis report
        analysis = []  # List to accumulate analysis text
        
        # Add header with total face count
        analysis.append(f"👥 Detected {face_stats['total_faces']} face(s) in the image\n")
        
        # Add detailed information for each face
        for face in face_stats.get('faces', []):
            # Create description for individual face
            face_desc = f"\n**Face {face['face_id']}:**"
            
            # Add position information
            pos = face.get('position', {})
            face_desc += f"\n  • Position: ({pos.get('x', 0)}, {pos.get('y', 0)})"
            
            # Add size information
            face_desc += f"\n  • Size: {pos.get('width', 0)}x{pos.get('height', 0)} pixels"
            
            # Add eye detection information if eyes were found
            if face.get('eyes_detected', 0) > 0:
                face_desc += f"\n  • Eyes detected: {face['eyes_detected']}"
            
            # Add smile detection information
            if face.get('smile_detected', False):
                face_desc += "\n  • 😊 Smile detected!"
            
            analysis.append(face_desc)  # Add face description to analysis
        
        # Add summary statistics if smiles were detected
        features = face_stats.get('features_detected', {})
        smile_count = features.get('smiles', 0)
        
        if smile_count > 0:
            # Calculate percentage of faces that are smiling
            smile_ratio = (smile_count / face_stats['total_faces']) * 100
            
            # Add overall analysis section
            analysis.append(f"\n\n📊 **Overall Analysis:**")
            analysis.append(f"\n  • {smile_ratio:.0f}% of faces are smiling")
            analysis.append(f"\n  • Total eyes detected: {features.get('eyes', 0)}")
        
        # Join all analysis parts and return
        return "".join(analysis)


# Create a global instance of FaceDetector for use throughout the application
# This avoids reloading classifiers multiple times
try:
    face_detector = FaceDetector()
    logger.info("Global face detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize global face detector: {str(e)}")
    # Create a dummy detector that returns empty results
    face_detector = None
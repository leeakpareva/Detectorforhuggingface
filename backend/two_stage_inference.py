"""
Two-Stage Inference System
Combines YOLO detection with custom classifier for improved accuracy
"""
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
import logging

from .yolo_enhanced import detect_objects_enhanced, model as yolo_model
from .custom_trainer import CustomClassifier

logger = logging.getLogger(__name__)

class TwoStageInference:
    """Two-stage detection and classification system"""
    
    def __init__(self, models_dir='models/'):
        """
        Initialize two-stage inference system
        
        Args:
            models_dir: Directory containing trained custom models
        """
        self.models_dir = Path(models_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load active custom model if available
        self.custom_model = None
        self.class_info = None
        self.load_active_model()
        
        # Image preprocessing for custom classifier
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_active_model(self):
        """Load the most recent trained custom model"""
        try:
            # Find latest model file
            model_files = list(self.models_dir.glob('custom_classifier_*.pkl'))
            if not model_files:
                logger.info("No custom models found. Using YOLO only.")
                return
            
            # Get most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # Load model info
            with open(latest_model, 'rb') as f:
                model_info = pickle.load(f)
            
            self.class_info = model_info['class_info']
            
            # Initialize and load custom model
            self.custom_model = CustomClassifier(
                num_classes=self.class_info['num_classes'],
                backbone=model_info['training_config']['backbone']
            )
            self.custom_model.load_state_dict(model_info['model_state'])
            self.custom_model = self.custom_model.to(self.device)
            self.custom_model.eval()
            
            logger.info(f"Loaded custom model: {latest_model.name}")
            logger.info(f"Custom classes: {list(self.class_info['idx_to_label'].values())}")
            
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            self.custom_model = None
            self.class_info = None
    
    def classify_object(self, object_crop: np.ndarray) -> Tuple[str, float]:
        """
        Classify object crop using custom model
        
        Args:
            object_crop: Cropped image region
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        if self.custom_model is None:
            return None, 0.0
        
        try:
            # Preprocess image
            if object_crop.size == 0:
                return None, 0.0
            
            # Convert BGR to RGB
            if len(object_crop.shape) == 3 and object_crop.shape[2] == 3:
                object_crop = cv2.cvtColor(object_crop, cv2.COLOR_BGR2RGB)
            
            # Preprocess for model
            input_tensor = self.preprocess(object_crop).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.custom_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_idx = predicted.item()
                confidence_score = confidence.item()
                
                # Convert to label
                predicted_label = self.class_info['idx_to_label'][predicted_idx]
                
                return predicted_label, confidence_score
            
        except Exception as e:
            logger.error(f"Custom classification failed: {e}")
            return None, 0.0
    
    def should_override_yolo(self, yolo_label: str, yolo_confidence: float,
                           custom_label: str, custom_confidence: float) -> bool:
        """
        Decide whether to override YOLO prediction with custom model
        
        Args:
            yolo_label: YOLO predicted label
            yolo_confidence: YOLO confidence
            custom_label: Custom model predicted label
            custom_confidence: Custom model confidence
            
        Returns:
            True if should use custom model prediction
        """
        # Don't override if custom model not confident enough
        if custom_confidence < 0.7:
            return False
        
        # Always override if YOLO has low confidence and custom has high
        if yolo_confidence < 0.5 and custom_confidence > 0.8:
            return True
        
        # Override if custom model is significantly more confident
        if custom_confidence > yolo_confidence + 0.2:
            return True
        
        # Override if we have training data for this custom class
        if custom_label in self.class_info.get('valid_classes', []):
            return True
        
        return False
    
    def detect_with_custom_model(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[np.ndarray, List[str], List[Dict]]:
        """
        Two-stage detection: YOLO + Custom Classification
        
        Args:
            image: Input image
            confidence_threshold: YOLO confidence threshold
            
        Returns:
            Tuple of (annotated_image, detected_objects, detailed_attributes)
        """
        # Stage 1: YOLO Detection
        try:
            annotated_img, detected_objects, detailed_attributes = detect_objects_enhanced(
                image, confidence_threshold
            )
        except:
            # Fallback to basic YOLO
            from .yolo import detect_objects
            annotated_img, detected_objects = detect_objects(image)
            detailed_attributes = []
        
        # Stage 2: Custom Classification (if model available)
        if self.custom_model is None or not detailed_attributes:
            return annotated_img, detected_objects, detailed_attributes
        
        # Process each detection with custom model
        enhanced_attributes = []
        enhanced_objects = []
        
        for i, attr in enumerate(detailed_attributes):
            yolo_label = attr['label']
            yolo_confidence = float(attr['confidence'].rstrip('%')) / 100.0
            bbox = attr.get('bbox', [0, 0, 100, 100])
            
            # Extract object region
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            object_crop = image[max(0, y1):min(image.shape[0], y2),
                              max(0, x1):min(image.shape[1], x2)]
            
            # Classify with custom model
            custom_label, custom_confidence = self.classify_object(object_crop)
            
            # Decide which prediction to use
            if custom_label and self.should_override_yolo(yolo_label, yolo_confidence, 
                                                        custom_label, custom_confidence):
                # Use custom model prediction
                final_label = custom_label
                final_confidence = custom_confidence
                attr['prediction_source'] = 'custom_model'
                attr['original_yolo'] = {'label': yolo_label, 'confidence': yolo_confidence}
            else:
                # Use YOLO prediction
                final_label = yolo_label
                final_confidence = yolo_confidence
                attr['prediction_source'] = 'yolo'
                if custom_label:
                    attr['custom_alternative'] = {'label': custom_label, 'confidence': custom_confidence}
            
            # Update attributes
            attr['label'] = final_label
            attr['confidence'] = f"{final_confidence:.2%}"
            
            enhanced_attributes.append(attr)
            enhanced_objects.append(final_label)
        
        # Update annotated image if we made changes
        if any(attr.get('prediction_source') == 'custom_model' for attr in enhanced_attributes):
            # Re-annotate image with updated predictions
            annotated_img = self.annotate_image_with_predictions(image, enhanced_attributes)
        
        return annotated_img, enhanced_objects, enhanced_attributes
    
    def annotate_image_with_predictions(self, image: np.ndarray, attributes: List[Dict]) -> np.ndarray:
        """
        Annotate image with updated predictions
        
        Args:
            image: Original image
            attributes: Detection attributes with updated labels
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for attr in attributes:
            bbox = attr.get('bbox', [0, 0, 100, 100])
            label = attr['label']
            confidence = attr['confidence']
            source = attr.get('prediction_source', 'yolo')
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Choose color based on source
            if source == 'custom_model':
                color = (0, 255, 0)  # Green for custom model
                label_text = f"{label} {confidence} (Custom)"
            else:
                color = (255, 0, 0)  # Red for YOLO
                label_text = f"{label} {confidence}"
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(annotated, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'yolo_model': 'YOLOv8m',
            'custom_model_loaded': self.custom_model is not None,
            'device': self.device
        }
        
        if self.custom_model is not None and self.class_info is not None:
            info.update({
                'custom_classes': list(self.class_info['idx_to_label'].values()),
                'num_custom_classes': self.class_info['num_classes'],
                'training_samples': self.class_info.get('train_samples', 0),
                'validation_samples': self.class_info.get('val_samples', 0)
            })
        
        return info

# Global two-stage inference instance
two_stage_inference = TwoStageInference()
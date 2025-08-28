"""
Advanced Recognition Module for NAVADA
Handles face recognition, custom object detection, and RAG-enhanced identification
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from .database import db
from .face_detection import face_detector
import time
import uuid

# Configure logging
logger = logging.getLogger(__name__)

class NAVADARecognition:
    """Advanced recognition system with database integration"""
    
    def __init__(self):
        """Initialize recognition system"""
        self.face_threshold = 0.6  # Face recognition threshold
        self.object_threshold = 0.5  # Object recognition threshold
        self.session_id = str(uuid.uuid4())
        
    def extract_face_encoding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face encoding for recognition
        This is a simplified version - in production, use face_recognition library
        """
        try:
            # Convert to grayscale and resize
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (128, 128))
            
            # Flatten and normalize as simple encoding
            encoding = resized.flatten().astype(np.float64)
            encoding = encoding / np.linalg.norm(encoding)  # Normalize
            
            return encoding
            
        except Exception as e:
            logger.error(f"Face encoding extraction failed: {e}")
            return None
    
    def compare_face_encodings(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Compare two face encodings and return similarity score"""
        try:
            # Calculate cosine similarity
            similarity = np.dot(encoding1, encoding2) / (
                np.linalg.norm(encoding1) * np.linalg.norm(encoding2)
            )
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return 0.0
    
    def recognize_faces(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Recognize faces in image against database
        
        Returns:
            Annotated image and list of recognition results
        """
        try:
            if not db:
                return image, []
            
            # Detect faces first
            annotated_img, face_stats = face_detector.detect_faces(image)
            
            # Get known faces from database
            known_faces = db.get_faces()
            
            recognition_results = []
            
            if face_stats and face_stats['faces']:
                for face_info in face_stats['faces']:
                    # Extract face region
                    pos = face_info['position']
                    x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
                    face_region = image[y:y+h, x:x+w]
                    
                    if face_region.size > 0:
                        # Extract face encoding
                        face_encoding = self.extract_face_encoding(face_region)
                        
                        if face_encoding is not None:
                            # Compare with known faces
                            best_match = None
                            best_similarity = 0.0
                            
                            for known_face in known_faces:
                                similarity = self.compare_face_encodings(
                                    face_encoding, known_face['encoding']
                                )
                                
                                if similarity > best_similarity and similarity > self.face_threshold:
                                    best_similarity = similarity
                                    best_match = known_face
                            
                            # Add recognition result
                            if best_match:
                                # Draw name on image
                                name = best_match['name']
                                cv2.putText(annotated_img, f"{name} ({best_similarity:.2f})", 
                                          (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                          (0, 255, 0), 2)
                                
                                recognition_results.append({
                                    'face_id': face_info['face_id'],
                                    'name': name,
                                    'similarity': best_similarity,
                                    'position': pos,
                                    'database_id': best_match['id']
                                })
                            else:
                                # Unknown face
                                cv2.putText(annotated_img, "Unknown", 
                                          (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                          (0, 0, 255), 2)
                                
                                recognition_results.append({
                                    'face_id': face_info['face_id'],
                                    'name': 'Unknown',
                                    'similarity': 0.0,
                                    'position': pos,
                                    'database_id': None
                                })
            
            return annotated_img, recognition_results
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return image, []
    
    def add_new_face(self, image: np.ndarray, name: str, face_region: Tuple = None) -> bool:
        """
        Add a new face to the database
        
        Args:
            image: Full image containing the face
            name: Person's name
            face_region: Optional (x, y, w, h) region, if None will detect automatically
            
        Returns:
            Success status
        """
        try:
            if not db:
                logger.error("Database not available")
                return False
            
            if face_region:
                # Use provided region
                x, y, w, h = face_region
                face_img = image[y:y+h, x:x+w]
            else:
                # Detect face automatically
                _, face_stats = face_detector.detect_faces(image)
                
                if not face_stats or not face_stats['faces']:
                    logger.error("No face detected in image")
                    return False
                
                # Use first detected face
                pos = face_stats['faces'][0]['position']
                x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
                face_img = image[y:y+h, x:x+w]
            
            # Extract encoding
            encoding = self.extract_face_encoding(face_img)
            if encoding is None:
                logger.error("Failed to extract face encoding")
                return False
            
            # Add to database
            face_id = db.add_face(
                name=name,
                face_encoding=encoding,
                image=face_img,
                confidence=0.9,
                metadata={
                    'source': 'user_added',
                    'session_id': self.session_id,
                    'timestamp': time.time()
                }
            )
            
            logger.info(f"Added new face '{name}' with ID {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add new face: {e}")
            return False
    
    def add_custom_object(self, image: np.ndarray, label: str, category: str, 
                         bbox: Tuple = None) -> bool:
        """
        Add a custom object to the database
        
        Args:
            image: Full image containing the object
            label: Object label/name
            category: Object category
            bbox: Optional (x, y, w, h) bounding box
            
        Returns:
            Success status
        """
        try:
            if not db:
                logger.error("Database not available")
                return False
            
            if bbox:
                # Use provided bounding box
                x, y, w, h = bbox
                object_img = image[y:y+h, x:x+w]
            else:
                # Use entire image as object
                object_img = image
                bbox = (0, 0, image.shape[1], image.shape[0])
            
            # Extract simple features (can be enhanced with deep learning)
            features = self.extract_object_features(object_img)
            
            # Add to database
            object_id = db.add_object(
                label=label,
                category=category,
                features=features,
                image=object_img,
                bounding_box=bbox,
                confidence=0.8,
                metadata={
                    'source': 'user_added',
                    'session_id': self.session_id,
                    'timestamp': time.time()
                }
            )
            
            logger.info(f"Added custom object '{label}' with ID {object_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add custom object: {e}")
            return False
    
    def extract_object_features(self, object_img: np.ndarray) -> np.ndarray:
        """Extract features from object image (simplified implementation)"""
        try:
            # Convert to grayscale and resize
            gray = cv2.cvtColor(object_img, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (64, 64))
            
            # Extract histogram features
            hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
            hist_normalized = hist.flatten() / hist.sum()
            
            # Extract edge features
            edges = cv2.Canny(resized, 50, 150)
            edge_density = edges.sum() / edges.size
            
            # Combine features
            features = np.concatenate([hist_normalized, [edge_density]])
            
            return features.astype(np.float64)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.array([])
    
    def enhance_with_rag(self, detections: List, face_matches: List = None) -> str:
        """
        Use RAG to enhance detection results with context
        
        Args:
            detections: List of detected objects
            face_matches: List of face recognition results
            
        Returns:
            Enhanced description with context
        """
        try:
            if not db:
                return "Enhanced analysis not available (database offline)"
            
            # Build search queries from detections
            queries = []
            
            # Add object queries
            for detection in detections:
                queries.append(detection)
            
            # Add face queries
            if face_matches:
                for match in face_matches:
                    if match['name'] != 'Unknown':
                        queries.append(match['name'])
            
            # Search knowledge base
            knowledge_results = []
            for query in queries:
                results = db.search_knowledge(query)
                knowledge_results.extend(results)
            
            # Build enhanced description
            if knowledge_results:
                enhanced_desc = "🧠 **Enhanced Analysis with Context:**\n\n"
                
                # Group by entity type
                face_context = [r for r in knowledge_results if r['entity_type'] == 'face']
                object_context = [r for r in knowledge_results if r['entity_type'] == 'object']
                
                if face_context:
                    enhanced_desc += "👥 **Known Individuals:**\n"
                    for ctx in face_context[:3]:  # Limit to 3 results
                        enhanced_desc += f"  • {ctx['content']}\n"
                    enhanced_desc += "\n"
                
                if object_context:
                    enhanced_desc += "🏷️ **Recognized Objects:**\n"
                    for ctx in object_context[:3]:  # Limit to 3 results
                        enhanced_desc += f"  • {ctx['content']}\n"
                    enhanced_desc += "\n"
                
                enhanced_desc += "📊 **Context Insights:**\n"
                enhanced_desc += f"  • Found {len(knowledge_results)} relevant knowledge entries\n"
                enhanced_desc += f"  • Analysis includes both detected and learned objects\n"
                
                return enhanced_desc
            else:
                return "🔍 **Context Analysis:** No additional context found in knowledge base."
                
        except Exception as e:
            logger.error(f"RAG enhancement failed: {e}")
            return "❌ Enhanced analysis unavailable due to processing error."
    
    def save_session_data(self, image: np.ndarray, detections: List, 
                         face_matches: List = None, processing_time: float = 0.0):
        """Save current session data to database"""
        try:
            if db:
                db.save_detection_history(
                    session_id=self.session_id,
                    image=image,
                    detections=detections,
                    face_matches=face_matches,
                    processing_time=processing_time,
                    metadata={
                        'timestamp': time.time(),
                        'version': '2.0'
                    }
                )
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")

# Global recognition instance
try:
    recognition_system = NAVADARecognition()
    logger.info("Recognition system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize recognition system: {e}")
    recognition_system = None
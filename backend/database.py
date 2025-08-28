"""
Database Module for NAVADA - SQLite storage for faces and objects
Handles storage, retrieval, and management of custom recognition data
"""

import sqlite3
import numpy as np
import cv2
from datetime import datetime
import json
import base64
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class NAVADADatabase:
    """Database manager for storing faces, objects, and recognition data"""
    
    def __init__(self, db_path: str = "navada_recognition.db"):
        """
        Initialize database connection and create tables
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create faces table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        encoding BLOB NOT NULL,
                        image_data BLOB,
                        confidence REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)
                
                # Create objects table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS objects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        label TEXT NOT NULL,
                        category TEXT,
                        features BLOB,
                        image_data BLOB,
                        bounding_box TEXT,
                        confidence REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        is_active BOOLEAN DEFAULT 1
                    )
                """)
                
                # Create detection history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detection_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        image_data BLOB,
                        detections TEXT,
                        face_matches TEXT,
                        object_matches TEXT,
                        confidence_scores TEXT,
                        processing_time REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # Create knowledge base for RAG
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_base (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_type TEXT NOT NULL,
                        entity_id INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding BLOB,
                        keywords TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create training corrections table for active learning
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS training_corrections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        image_path TEXT,
                        image_crop BLOB NOT NULL,
                        bbox_coords TEXT NOT NULL,
                        yolo_prediction TEXT NOT NULL,
                        yolo_confidence REAL NOT NULL,
                        correct_label TEXT NOT NULL,
                        user_feedback TEXT,
                        difficulty_score REAL DEFAULT 0.0,
                        validated BOOLEAN DEFAULT 0,
                        used_for_training BOOLEAN DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        session_id TEXT,
                        metadata TEXT
                    )
                """)
                
                # Create custom model versions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version_name TEXT NOT NULL UNIQUE,
                        model_path TEXT NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        training_samples INTEGER DEFAULT 0,
                        validation_samples INTEGER DEFAULT 0,
                        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 0,
                        performance_metrics TEXT,
                        training_config TEXT,
                        notes TEXT
                    )
                """)
                
                # Create custom classes mapping
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS custom_classes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        class_name TEXT NOT NULL UNIQUE,
                        yolo_class TEXT,
                        sample_count INTEGER DEFAULT 0,
                        confidence_threshold REAL DEFAULT 0.5,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        description TEXT
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_objects_label ON objects(label)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_session ON detection_history(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_entity ON knowledge_base(entity_type, entity_id)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def add_face(self, name: str, face_encoding: np.ndarray, image: np.ndarray, 
                 confidence: float = 0.0, metadata: Dict = None) -> int:
        """
        Add a new face to the database
        
        Args:
            name: Person's name
            face_encoding: Face encoding vector
            image: Face image array
            confidence: Recognition confidence
            metadata: Additional metadata
            
        Returns:
            Face ID in database
        """
        try:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', image)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Serialize face encoding
            encoding_data = face_encoding.tobytes()
            
            # Convert metadata to JSON
            metadata_json = json.dumps(metadata) if metadata else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO faces (name, encoding, image_data, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, encoding_data, image_data, confidence, metadata_json))
                
                face_id = cursor.lastrowid
                conn.commit()
                
                # Add to knowledge base
                self.add_knowledge_entry("face", face_id, f"Person named {name}")
                
                logger.info(f"Added face for {name} with ID {face_id}")
                return face_id
                
        except Exception as e:
            logger.error(f"Failed to add face: {e}")
            raise
    
    def add_object(self, label: str, category: str, features: np.ndarray, 
                   image: np.ndarray, bounding_box: Tuple, confidence: float = 0.0, 
                   metadata: Dict = None) -> int:
        """
        Add a new custom object to the database
        
        Args:
            label: Object label/name
            category: Object category
            features: Feature vector
            image: Object image
            bounding_box: (x, y, w, h) bounding box
            confidence: Detection confidence
            metadata: Additional metadata
            
        Returns:
            Object ID in database
        """
        try:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', image)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Serialize features
            features_data = features.tobytes() if features is not None else None
            
            # Serialize bounding box
            bbox_json = json.dumps(bounding_box)
            
            # Convert metadata to JSON
            metadata_json = json.dumps(metadata) if metadata else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO objects (label, category, features, image_data, 
                                       bounding_box, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (label, category, features_data, image_data, bbox_json, 
                      confidence, metadata_json))
                
                object_id = cursor.lastrowid
                conn.commit()
                
                # Add to knowledge base
                self.add_knowledge_entry("object", object_id, 
                                       f"{label} - {category} object")
                
                logger.info(f"Added object {label} with ID {object_id}")
                return object_id
                
        except Exception as e:
            logger.error(f"Failed to add object: {e}")
            raise
    
    def get_faces(self, active_only: bool = True) -> List[Dict]:
        """Get all faces from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM faces"
                if active_only:
                    query += " WHERE is_active = 1"
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                faces = []
                for row in rows:
                    face = {
                        'id': row[0],
                        'name': row[1],
                        'encoding': np.frombuffer(row[2], dtype=np.float64),
                        'confidence': row[4],
                        'created_at': row[5],
                        'metadata': json.loads(row[7]) if row[7] else {}
                    }
                    faces.append(face)
                
                return faces
                
        except Exception as e:
            logger.error(f"Failed to get faces: {e}")
            return []
    
    def get_objects(self, category: str = None, active_only: bool = True) -> List[Dict]:
        """Get objects from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                query = "SELECT * FROM objects"
                params = []
                
                conditions = []
                if active_only:
                    conditions.append("is_active = 1")
                if category:
                    conditions.append("category = ?")
                    params.append(category)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                objects = []
                for row in rows:
                    obj = {
                        'id': row[0],
                        'label': row[1],
                        'category': row[2],
                        'features': np.frombuffer(row[3], dtype=np.float64) if row[3] else None,
                        'bounding_box': json.loads(row[5]) if row[5] else None,
                        'confidence': row[6],
                        'created_at': row[7],
                        'metadata': json.loads(row[9]) if row[9] else {}
                    }
                    objects.append(obj)
                
                return objects
                
        except Exception as e:
            logger.error(f"Failed to get objects: {e}")
            return []
    
    def save_detection_history(self, session_id: str, image: np.ndarray, 
                             detections: List, face_matches: List = None, 
                             object_matches: List = None, confidence_scores: Dict = None,
                             processing_time: float = 0.0, metadata: Dict = None) -> int:
        """Save detection results to history"""
        try:
            # Encode image
            _, buffer = cv2.imencode('.jpg', image)
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Serialize data
            detections_json = json.dumps(detections)
            face_matches_json = json.dumps(face_matches) if face_matches else None
            object_matches_json = json.dumps(object_matches) if object_matches else None
            confidence_json = json.dumps(confidence_scores) if confidence_scores else None
            metadata_json = json.dumps(metadata) if metadata else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO detection_history 
                    (session_id, image_data, detections, face_matches, object_matches,
                     confidence_scores, processing_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, image_data, detections_json, face_matches_json,
                      object_matches_json, confidence_json, processing_time, metadata_json))
                
                history_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Saved detection history with ID {history_id}")
                return history_id
                
        except Exception as e:
            logger.error(f"Failed to save detection history: {e}")
            raise
    
    def add_knowledge_entry(self, entity_type: str, entity_id: int, content: str, 
                          keywords: List[str] = None):
        """Add entry to knowledge base for RAG"""
        try:
            keywords_json = json.dumps(keywords) if keywords else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO knowledge_base (entity_type, entity_id, content, keywords)
                    VALUES (?, ?, ?, ?)
                """, (entity_type, entity_id, content, keywords_json))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to add knowledge entry: {e}")
    
    def search_knowledge(self, query: str, entity_type: str = None) -> List[Dict]:
        """Search knowledge base for RAG"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Simple text search (can be enhanced with embeddings)
                search_query = f"%{query.lower()}%"
                
                if entity_type:
                    cursor.execute("""
                        SELECT * FROM knowledge_base 
                        WHERE entity_type = ? AND LOWER(content) LIKE ?
                        ORDER BY created_at DESC LIMIT 10
                    """, (entity_type, search_query))
                else:
                    cursor.execute("""
                        SELECT * FROM knowledge_base 
                        WHERE LOWER(content) LIKE ?
                        ORDER BY created_at DESC LIMIT 10
                    """, (search_query,))
                
                rows = cursor.fetchall()
                results = []
                
                for row in rows:
                    result = {
                        'id': row[0],
                        'entity_type': row[1],
                        'entity_id': row[2],
                        'content': row[3],
                        'keywords': json.loads(row[5]) if row[5] else [],
                        'created_at': row[6]
                    }
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count faces
                cursor.execute("SELECT COUNT(*) FROM faces WHERE is_active = 1")
                face_count = cursor.fetchone()[0]
                
                # Count objects
                cursor.execute("SELECT COUNT(*) FROM objects WHERE is_active = 1")
                object_count = cursor.fetchone()[0]
                
                # Count history entries
                cursor.execute("SELECT COUNT(*) FROM detection_history")
                history_count = cursor.fetchone()[0]
                
                # Get recent activity
                cursor.execute("""
                    SELECT COUNT(*) FROM detection_history 
                    WHERE created_at > datetime('now', '-7 days')
                """)
                recent_detections = cursor.fetchone()[0]
                
                return {
                    'faces': face_count,
                    'objects': object_count,
                    'total_detections': history_count,
                    'recent_detections': recent_detections,
                    'database_size': Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    # Training Corrections Methods for Active Learning
    
    def save_correction(self, image_crop: np.ndarray, bbox_coords: List[float], 
                       yolo_prediction: str, yolo_confidence: float, 
                       correct_label: str, user_feedback: str = "", 
                       session_id: str = "") -> bool:
        """
        Save a user correction for training
        
        Args:
            image_crop: Cropped image of the detected object
            bbox_coords: [x1, y1, x2, y2] bounding box coordinates
            yolo_prediction: Original YOLO predicted label
            yolo_confidence: Original YOLO confidence score
            correct_label: User-provided correct label
            user_feedback: Optional user feedback text
            session_id: Session identifier
            
        Returns:
            bool: Success status
        """
        try:
            # Convert image to bytes
            _, buffer = cv2.imencode('.jpg', image_crop)
            image_bytes = buffer.tobytes()
            
            # Calculate difficulty score (lower confidence = higher difficulty)
            difficulty_score = 1.0 - yolo_confidence
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO training_corrections 
                    (image_crop, bbox_coords, yolo_prediction, yolo_confidence, 
                     correct_label, user_feedback, difficulty_score, session_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_bytes,
                    json.dumps(bbox_coords),
                    yolo_prediction,
                    yolo_confidence,
                    correct_label,
                    user_feedback,
                    difficulty_score,
                    session_id,
                    json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'image_shape': image_crop.shape,
                        'correction_type': 'user_feedback'
                    })
                ))
                
                # Update or create custom class entry
                cursor.execute("""
                    INSERT OR IGNORE INTO custom_classes (class_name, yolo_class, sample_count)
                    VALUES (?, ?, 0)
                """, (correct_label, yolo_prediction))
                
                cursor.execute("""
                    UPDATE custom_classes 
                    SET sample_count = sample_count + 1 
                    WHERE class_name = ?
                """, (correct_label,))
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to save correction: {e}")
            return False
    
    def get_training_data(self, class_name: str = None, limit: int = 1000, 
                         validated_only: bool = False) -> List[Dict]:
        """
        Retrieve training data for model training
        
        Args:
            class_name: Filter by specific class (optional)
            limit: Maximum number of samples to return
            validated_only: Only return validated corrections
            
        Returns:
            List of training samples
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT id, image_crop, bbox_coords, yolo_prediction, 
                           yolo_confidence, correct_label, difficulty_score, 
                           created_at, metadata
                    FROM training_corrections 
                    WHERE 1=1
                """
                params = []
                
                if class_name:
                    query += " AND correct_label = ?"
                    params.append(class_name)
                
                if validated_only:
                    query += " AND validated = 1"
                
                query += " ORDER BY difficulty_score DESC, created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                training_data = []
                for row in rows:
                    # Decode image
                    image_bytes = row[1]
                    image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                    training_data.append({
                        'id': row[0],
                        'image': image_array,
                        'bbox_coords': json.loads(row[2]),
                        'yolo_prediction': row[3],
                        'yolo_confidence': row[4],
                        'correct_label': row[5],
                        'difficulty_score': row[6],
                        'created_at': row[7],
                        'metadata': json.loads(row[8]) if row[8] else {}
                    })
                
                return training_data
                
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []
    
    def get_training_stats(self) -> Dict:
        """Get statistics about training corrections"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total corrections
                cursor.execute("SELECT COUNT(*) FROM training_corrections")
                total_corrections = cursor.fetchone()[0]
                
                # Corrections by class
                cursor.execute("""
                    SELECT correct_label, COUNT(*) as count
                    FROM training_corrections 
                    GROUP BY correct_label 
                    ORDER BY count DESC
                """)
                class_counts = dict(cursor.fetchall())
                
                # Validated corrections
                cursor.execute("SELECT COUNT(*) FROM training_corrections WHERE validated = 1")
                validated_count = cursor.fetchone()[0]
                
                # Recent corrections (last 7 days)
                cursor.execute("""
                    SELECT COUNT(*) FROM training_corrections 
                    WHERE created_at > datetime('now', '-7 days')
                """)
                recent_corrections = cursor.fetchone()[0]
                
                # Average difficulty score
                cursor.execute("SELECT AVG(difficulty_score) FROM training_corrections")
                avg_difficulty = cursor.fetchone()[0] or 0.0
                
                return {
                    'total_corrections': total_corrections,
                    'validated_corrections': validated_count,
                    'recent_corrections': recent_corrections,
                    'class_distribution': class_counts,
                    'average_difficulty': round(avg_difficulty, 3),
                    'unique_classes': len(class_counts)
                }
                
        except Exception as e:
            logger.error(f"Failed to get training stats: {e}")
            return {}
    
    def mark_corrections_used(self, correction_ids: List[int]) -> bool:
        """Mark corrections as used for training"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                placeholders = ','.join(['?'] * len(correction_ids))
                cursor.execute(f"""
                    UPDATE training_corrections 
                    SET used_for_training = 1 
                    WHERE id IN ({placeholders})
                """, correction_ids)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to mark corrections as used: {e}")
            return False

# Global database instance
try:
    db = NAVADADatabase()
    logger.info("Database instance created successfully")
except Exception as e:
    logger.error(f"Failed to create database instance: {e}")
    db = None
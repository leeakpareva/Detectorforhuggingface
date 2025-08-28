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

# Global database instance
try:
    db = NAVADADatabase()
    logger.info("Database instance created successfully")
except Exception as e:
    logger.error(f"Failed to create database instance: {e}")
    db = None
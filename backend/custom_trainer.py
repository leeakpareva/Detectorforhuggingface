"""
Custom Object Classifier Training Module
Implements transfer learning for user feedback corrections
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class CustomObjectDataset(Dataset):
    """Dataset class for custom object training"""
    
    def __init__(self, data: List[Dict], transform=None):
        """
        Initialize dataset with training data
        
        Args:
            data: List of training samples from database
            transform: Image transformations
        """
        self.data = data
        self.transform = transform
        
        # Create label mapping
        unique_labels = list(set([sample['correct_label'] for sample in data]))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image']
        label = sample['correct_label']
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.label_to_idx[label]
        
        return {
            'image': image,
            'label': label_idx,
            'original_label': label,
            'yolo_prediction': sample['yolo_prediction'],
            'confidence': sample['yolo_confidence'],
            'difficulty': sample['difficulty_score']
        }

class CustomClassifier(nn.Module):
    """Custom classifier built on pre-trained backbone"""
    
    def __init__(self, num_classes: int, backbone='resnet18', pretrained=True):
        """
        Initialize custom classifier
        
        Args:
            num_classes: Number of output classes
            backbone: Backbone architecture (resnet18, resnet50, efficientnet_b0)
            pretrained: Use pre-trained weights
        """
        super(CustomClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.backbone = backbone
        
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif backbone == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x):
        return self.model(x)

class CustomTrainer:
    """Trainer class for custom object classification"""
    
    def __init__(self, model_dir='models/', device=None):
        """
        Initialize trainer
        
        Args:
            model_dir: Directory to save models
            device: Training device (cuda/cpu)
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Image transformations
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_data(self, training_data: List[Dict], test_size=0.2, min_samples_per_class=5):
        """
        Prepare training and validation data
        
        Args:
            training_data: List of training samples from database
            test_size: Fraction for validation split
            min_samples_per_class: Minimum samples required per class
            
        Returns:
            Tuple of (train_dataset, val_dataset, class_info)
        """
        # Filter classes with insufficient samples
        class_counts = {}
        for sample in training_data:
            label = sample['correct_label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Remove classes with insufficient samples
        valid_classes = {label for label, count in class_counts.items() 
                        if count >= min_samples_per_class}
        
        filtered_data = [sample for sample in training_data 
                        if sample['correct_label'] in valid_classes]
        
        if len(filtered_data) < 10:
            raise ValueError(f"Insufficient training data: {len(filtered_data)} samples")
        
        if len(valid_classes) < 2:
            raise ValueError(f"Need at least 2 classes, got {len(valid_classes)}")
        
        # Split data
        train_data, val_data = train_test_split(
            filtered_data, 
            test_size=test_size, 
            stratify=[sample['correct_label'] for sample in filtered_data],
            random_state=42
        )
        
        # Create datasets
        train_dataset = CustomObjectDataset(train_data, self.train_transform)
        val_dataset = CustomObjectDataset(val_data, self.val_transform)
        
        # Ensure same label mapping
        val_dataset.label_to_idx = train_dataset.label_to_idx
        val_dataset.idx_to_label = train_dataset.idx_to_label
        val_dataset.num_classes = train_dataset.num_classes
        
        class_info = {
            'num_classes': train_dataset.num_classes,
            'label_to_idx': train_dataset.label_to_idx,
            'idx_to_label': train_dataset.idx_to_label,
            'class_counts': class_counts,
            'valid_classes': list(valid_classes),
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        }
        
        return train_dataset, val_dataset, class_info
    
    def train_model(self, training_data: List[Dict], 
                   epochs=20, batch_size=16, learning_rate=0.001,
                   backbone='resnet18', patience=5) -> Dict:
        """
        Train custom classifier
        
        Args:
            training_data: Training samples from database
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            backbone: Model backbone architecture
            patience: Early stopping patience
            
        Returns:
            Training results and metrics
        """
        try:
            # Prepare data
            train_dataset, val_dataset, class_info = self.prepare_data(training_data)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            model = CustomClassifier(class_info['num_classes'], backbone)
            model = model.to(self.device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            
            # Training history
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
            
            best_val_acc = 0.0
            patience_counter = 0
            
            logger.info(f"Starting training: {epochs} epochs, {class_info['num_classes']} classes")
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch in train_loader:
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                train_acc = train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch['image'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_acc = val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)
                
                # Update history
                history['train_loss'].append(avg_train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_acc)
                
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), self.model_dir / 'best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                scheduler.step()
            
            # Load best model
            model.load_state_dict(torch.load(self.model_dir / 'best_model.pth'))
            
            # Final evaluation
            final_metrics = self.evaluate_model(model, val_loader, class_info)
            
            # Save model and metadata
            model_info = {
                'model_state': model.state_dict(),
                'class_info': class_info,
                'training_config': {
                    'backbone': backbone,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                },
                'history': history,
                'metrics': final_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save complete model info
            model_path = self.model_dir / f'custom_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model_info, f)
            
            logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
            logger.info(f"Model saved to: {model_path}")
            
            return {
                'success': True,
                'model_path': str(model_path),
                'best_accuracy': best_val_acc,
                'final_metrics': final_metrics,
                'class_info': class_info,
                'history': history
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def evaluate_model(self, model, val_loader, class_info) -> Dict:
        """Evaluate model performance"""
        model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label']
                
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_confidence': float(np.mean(all_confidences)),
            'num_samples': len(all_labels)
        }

# Global trainer instance
custom_trainer = CustomTrainer()
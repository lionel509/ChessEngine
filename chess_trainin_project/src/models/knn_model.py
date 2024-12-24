import logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import Tuple, Dict, Any
import time
import psutil
import sys
from .base_model import ChessModel  # Updated import path

class KNNChessModel(ChessModel):
    def __init__(self, n_neighbors: int = 5, weights: str = 'distance'):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Add file handler
        fh = logging.FileHandler('knn_model_debug.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.debug(f"Initializing KNN model with n_neighbors={n_neighbors}, weights={weights}")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric='euclidean'
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _preprocess_board(self, board_state: np.ndarray) -> np.ndarray:
        """Convert board state to feature vector"""
        if not self._validate_board_state(board_state):
            raise ValueError("Invalid board state")
        return board_state.flatten().reshape(1, -1)

    def predict_move(self, board_state: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Predict best move for given board state"""
        if not self.is_trained:
            self.logger.error("Model not trained before prediction")
            raise ValueError("Model must be trained before prediction")

        # Preprocess and scale board state
        X = self._preprocess_board(board_state)
        X_scaled = self.scaler.transform(X)

        # Get model prediction
        move_encoded = self.model.predict(X_scaled)[0]
        return self._decode_move(move_encoded)

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        start_time = time.time()
        self.logger.debug(f"Starting training with data shapes: X={X.shape}, y={y.shape}")
        self.logger.debug(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        
        if len(X.shape) == 4:
            self.logger.debug(f"Reshaping 4D input: {X.shape}")
            X = X.reshape(X.shape[0], -1)
            self.logger.debug(f"Reshaped to: {X.shape}")
            
        self.logger.debug(f"Data statistics - Mean: {X.mean():.4f}, Std: {X.std():.4f}")
        self.logger.debug(f"Label distribution: {np.unique(y, return_counts=True)}")
        
        # Scale features
        self.logger.debug("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        self.logger.debug(f"Scaled data statistics - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
        
        # Train model
        self.logger.debug("Fitting KNN model...")
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate metrics
        train_accuracy = self.model.score(X_scaled, y)
        self.logger.debug(f"Training accuracy: {train_accuracy:.4f}")
        
        training_time = time.time() - start_time
        self.logger.debug(f"Training completed in {training_time:.2f} seconds")
        self.logger.debug(f"Final memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        
        return {
            'train_accuracy': train_accuracy,
            'training_time': training_time,
            'data_shape': X.shape,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        start_time = time.time()
        self.logger.debug(f"Starting prediction on data shape: {X.shape}")
        
        if not self.is_trained:
            self.logger.error("Model not trained before prediction")
            raise ValueError("Model must be trained before prediction")
            
        if len(X.shape) == 4:
            self.logger.debug(f"Reshaping prediction input from {X.shape}")
            X = X.reshape(X.shape[0], -1)
            self.logger.debug(f"Reshaped to {X.shape}")
            
        X_scaled = self.scaler.transform(X)
        self.logger.debug(f"Scaled prediction data - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
        
        predictions = self.model.predict(X_scaled)
        self.logger.debug(f"Prediction distribution: {np.unique(predictions, return_counts=True)}")
        
        pred_time = time.time() - start_time
        self.logger.debug(f"Predictions completed in {pred_time:.2f} seconds")
        
        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        start_time = time.time()
        self.logger.debug(f"Starting evaluation on data shapes: X={X.shape}, y={y.shape}")
        
        if not self.is_trained:
            self.logger.error("Model not trained before evaluation")
            raise ValueError("Model must be trained before evaluation")
            
        if len(X.shape) == 4:
            self.logger.debug(f"Reshaping evaluation input from {X.shape}")
            X = X.reshape(X.shape[0], -1)
            self.logger.debug(f"Reshaped to {X.shape}")
            
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'evaluation_time': time.time() - start_time
        }
        
        self.logger.debug("Evaluation metrics:")
        for metric, value in metrics.items():
            self.logger.debug(f"{metric}: {value}")
            
        return metrics

    def get_neighbors(self, board_state: np.ndarray, k: int = None) -> np.ndarray:
        """Get k nearest neighbor positions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting neighbors")

        k = k or self.n_neighbors
        X = self._preprocess_board(board_state)
        X_scaled = self.scaler.transform(X)
        
        distances, indices = self.model.kneighbors(
            X_scaled, 
            n_neighbors=k, 
            return_distance=True
        )
        
        return indices[0], distances[0]

    def get_model_params(self) -> Dict[str, Any]:
        """Return model parameters"""
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.model.metric,
            'is_trained': self.is_trained
        }
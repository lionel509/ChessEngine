import os
import joblib
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import deque
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.knn_model import KNNChessModel

class ModelEnsemble:
    def __init__(self, models: List[Any]):
        self.models = models
        self.move_history = deque(maxlen=10)  # Track last 10 moves
        self.position_history = {}  # Track position frequencies
        
    def predict(self, board_state: np.ndarray) -> Tuple[int, int]:
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(board_state.reshape(1, -1))
            predictions.append(pred)
        
        # Convert predictions to moves
        moves = [self._decode_move(p[0]) for p in predictions]
        
        # Score moves based on history
        scored_moves = []
        pos_key = self._get_position_key(board_state)
        
        for move in moves:
            score = self._score_move(move, pos_key)
            scored_moves.append((score, move))
        
        # Select best non-repetitive move
        scored_moves.sort(reverse=True)  # Higher scores are better
        selected_move = scored_moves[0][1]
        
        # Update history
        self.move_history.append(selected_move)
        self.position_history[pos_key] = self.position_history.get(pos_key, 0) + 1
        
        return selected_move
    
    def _score_move(self, move: Tuple[int, int], pos_key: str) -> float:
        base_score = 1.0
        
        # Penalize repeated positions
        pos_repeat_penalty = self.position_history.get(pos_key, 0) * 0.2
        
        # Penalize moves that were recently played
        recent_move_penalty = 0.1 if move in self.move_history else 0
        
        return base_score - pos_repeat_penalty - recent_move_penalty
    
    def _get_position_key(self, board_state: np.ndarray) -> str:
        return hash(board_state.tobytes())
    
    def _decode_move(self, move_encoded: int) -> Tuple[int, int]:
        start_square = move_encoded // 64
        end_square = move_encoded % 64
        return (start_square, end_square)

def find_pkl_files(directory):
    """Recursively find all .pkl files in directory"""
    pkl_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def load_models(model_directory: str) -> Dict[str, Any]:
    logging.info(f"Loading models from {model_directory}")
    models = {}
    
    try:
        meta_path = Path(model_directory) / 'ensemble_meta.joblib'
        if not meta_path.exists():
            logging.error(f"Metadata file not found: {meta_path}")
            return {}
            
        meta = joblib.load(meta_path)
        ensemble_models = []
        
        for model_file in meta['model_files']:
            model_path = Path(model_directory) / model_file
            if model_path.exists():
                model = joblib.load(model_path)
                ensemble_models.append(model)
                logging.info(f"Loaded model: {model_file}")
                
        if ensemble_models:
            models['knn_ensemble'] = ModelEnsemble(ensemble_models)
            logging.info(f"Created ensemble with {len(ensemble_models)} models")
            
        return models
        
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        return {}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_dir = r"C:\Users\Lionel\Downloads\ChessEngine\models"
    loaded_models = load_models(model_dir)
    print(f"Loaded {len(loaded_models)} model ensembles")
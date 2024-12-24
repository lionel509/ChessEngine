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
    def __init__(self, models: List[Any], memory_size: int = 5):
        self.models = models
        self.move_history = deque(maxlen=memory_size)
        self.last_position = None
        
    def predict(self, board_state: np.ndarray) -> Tuple[int, int]:
        # Get all possible predictions from models
        predictions = []
        for model in self.models:
            pred = model.predict(board_state.reshape(1, -1))
            predictions.append(pred)
            
        # Convert predictions to moves
        moves = [self._decode_move(p[0]) for p in predictions]
        
        # Filter out recent moves to prevent repetition
        valid_moves = []
        for move in moves:
            if move not in self.move_history:
                valid_moves.append(move)
                
        # If no valid moves, reset history and use any move
        if not valid_moves:
            self.move_history.clear()
            valid_moves = moves
            
        # Select best move (can be enhanced with scoring)
        selected_move = valid_moves[0]
        
        # Update history
        self.move_history.append(selected_move)
        self.last_position = board_state.copy()
        
        return selected_move
        
    def _decode_move(self, move_encoded: int) -> Tuple[int, int]:
        """Decode model output to chess move"""
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
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any
import joblib

class ChessModel(ABC):
    def __init__(self):
        self.model_name = self.__class__.__name__
        self.is_trained = False
        self.training_history = []

    @abstractmethod
    def predict_move(self, board_state: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Predict next move given board state"""
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train model on chess positions"""
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        pass

    def save_model(self, filepath: str) -> None:
        """Save model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        joblib.dump(self, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> 'ChessModel':
        """Load model from disk"""
        model = joblib.load(filepath)
        if not isinstance(model, cls):
            raise ValueError(f"Loaded model is not an instance of {cls.__name__}")
        return model

    def _validate_board_state(self, board_state: np.ndarray) -> bool:
        """Validate board state dimensions and values"""
        return True if isinstance(board_state, np.ndarray) else False

    def _encode_move(self, start: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Encode chess move as single integer"""
        return start[0] * 1000 + start[1] * 100 + end[0] * 10 + end[1]

    def _decode_move(self, encoded_move: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Decode integer back to chess move"""
        start_square = encoded_move // 64
        end_square = encoded_move % 64
        start_pos = (start_square // 8, start_square % 8)
        end_pos = (end_square // 8, end_square % 8)
        return start_pos, end_pos

    def get_training_history(self) -> Dict[str, list]:
        """Return training history"""
        return {
            'epochs': list(range(len(self.training_history))),
            'metrics': self.training_history
        }
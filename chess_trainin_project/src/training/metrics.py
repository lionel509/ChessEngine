import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

class ChessMetrics:
    def __init__(self):
        self.reset_metrics()

    def reset_metrics(self):
        """Reset all metrics"""
        self.game_outcomes = defaultdict(int)
        self.move_accuracy = []
        self.position_scores = []
        self.elo_ratings = {}
        self.game_lengths = []

    def calculate_move_accuracy(self, predicted_move: Tuple[int, int], actual_move: Tuple[int, int]) -> float:
        """Calculate accuracy of predicted move"""
        start_accuracy = 1.0 if predicted_move[0] == actual_move[0] else 0.0
        end_accuracy = 1.0 if predicted_move[1] == actual_move[1] else 0.0
        return (start_accuracy + end_accuracy) / 2.0

    def update_game_outcome(self, result: str):
        """Update game outcomes counter"""
        self.game_outcomes[result] += 1
        
    def add_position_score(self, score: float):
        """Add evaluation score for a position"""
        self.position_scores.append(score)
        
    def update_elo_rating(self, player_id: str, new_rating: int):
        """Update Elo rating for a player"""
        self.elo_ratings[player_id] = new_rating
        
    def add_game_length(self, moves: int):
        """Track number of moves in completed game"""
        self.game_lengths.append(moves)
        
    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics"""
        stats = {
            'total_games': sum(self.game_outcomes.values()),
            'win_rate': self.game_outcomes['win'] / max(sum(self.game_outcomes.values()), 1),
            'avg_move_accuracy': np.mean(self.move_accuracy) if self.move_accuracy else 0.0,
            'avg_position_score': np.mean(self.position_scores) if self.position_scores else 0.0,
            'avg_game_length': np.mean(self.game_lengths) if self.game_lengths else 0.0
        }
        return stats

    def get_all_metrics(self) -> Dict:
        """Get all metrics as dictionary"""
        return {
            'game_outcomes': dict(self.game_outcomes),
            'move_accuracy': self.move_accuracy,
            'position_scores': self.position_scores,
            'elo_ratings': self.elo_ratings,
            'game_lengths': self.game_lengths,
            'summary': self.get_summary_stats()
        }



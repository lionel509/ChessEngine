import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List, Optional
from .board import ChessBoard

class Player:
    def __init__(self, color: str):
        self.color = color  # 'white' or 'black'
        self.game_history = []

    def get_move(self, board: ChessBoard) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        raise NotImplementedError("Subclasses must implement get_move")

    def update_history(self, board_state: np.ndarray, move: Tuple[Tuple[int, int], Tuple[int, int]]):
        self.game_history.append((board_state.copy(), move))

class KNNPlayer(Player):
    def __init__(self, color: str, model_path: Optional[str] = None):
        super().__init__(color)
        self.model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        self.trained = False
        if model_path:
            self.load_model(model_path)

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the KNN model on chess positions"""
        self.model.fit(X.reshape(X.shape[0], -1), y)
        self.trained = True

    def get_move(self, board: ChessBoard) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Select best move using KNN model"""
        if not self.trained:
            raise ValueError("Model must be trained before making moves")

        # Get current board state
        current_state = board.get_board_state()
        
        # Get all legal moves for current player's pieces
        legal_moves = []
        for i in range(8):
            for j in range(8):
                piece = current_state[i][j]
                if (self.color == 'white' and piece > 0) or \
                   (self.color == 'black' and piece < 0):
                    moves = board.get_legal_moves((i, j))
                    legal_moves.extend([((i, j), move) for move in moves])

        if not legal_moves:
            return None  # No legal moves available

        # Predict best move
        move_scores = []
        for start, end in legal_moves:
            # Create hypothetical board state
            test_board = current_state.copy()
            test_board[end[0]][end[1]] = test_board[start[0]][start[1]]
            test_board[start[0]][start[1]] = 0
            
            # Get model prediction score
            score = self.model.predict_proba(test_board.reshape(1, -1))[0]
            move_scores.append((score, (start, end)))

        # Select move with highest score
        best_move = max(move_scores, key=lambda x: x[0])[1]
        self.update_history(current_state, best_move)
        return best_move

    def save_model(self, path: str):
        """Save trained model"""
        import joblib
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load trained model"""
        import joblib
        self.model = joblib.load(path)
        self.trained = True

class HumanPlayer(Player):
    def get_move(self, board: ChessBoard) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get move from human input"""
        while True:
            try:
                move = input(f"{self.color}'s move (e.g. e2e4): ")
                if len(move) != 4:
                    raise ValueError
                
                # Convert chess notation to coordinates
                start = (int(move[1]) - 1, ord(move[0]) - ord('a'))
                end = (int(move[3]) - 1, ord(move[2]) - ord('a'))
                
                if board.is_valid_move(start, end):
                    self.update_history(board.get_board_state(), (start, end))
                    return start, end
                    
            except ValueError:
                print("Invalid move format. Please use format 'e2e4'")
            except IndexError:
                print("Invalid move format. Please use format 'e2e4'")
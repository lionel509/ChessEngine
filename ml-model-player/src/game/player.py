import chess
import chess.svg
from models.loader import load_models

class Player:
    def __init__(self, models):
        self.models = models
        self.board = chess.Board()
        
    def play_game(self):
        print("\nAvailable models:")
        for i, model_name in enumerate(self.models.keys(), 1):
            print(f"{i}. {model_name}")
            
        model_idx = int(input("\nSelect model to play against (number): ")) - 1
        model_name = list(self.models.keys())[model_idx]
        model = self.models[model_name]
        
        print(f"\nPlaying against {model_name}")
        print("Enter moves in UCI format (e.g. e2e4)")
        
        while not self.board.is_game_over():
            print(f"\n{self.board}")
            
            if self.board.turn == chess.WHITE:
                # Human move
                while True:
                    try:
                        move = input("\nYour move: ")
                        self.board.push_uci(move)
                        break
                    except ValueError:
                        print("Invalid move, try again")
            else:
                # AI move
                legal_moves = list(self.board.legal_moves)
                move = legal_moves[0]  # Default to first legal move
                print(f"\nAI plays: {move}")
                self.board.push(move)
                
        print(f"\nGame Over! Result: {self.board.result()}")

def encode_board_state(board):
    """Convert board state to model input format"""
    # Implement based on your model's expected input format
    pass

def decode_move(move_idx):
    """Convert model output to chess move"""
    # Implement based on your model's output format
    pass
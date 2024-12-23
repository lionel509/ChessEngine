import numpy as np
import chess
import pandas as pd
import random

def load_training_data(file_path):
    data = pd.read_csv(file_path)
    X = np.array([eval(row["board_state"]) for _, row in data.iterrows()])
    y = np.array([eval(row["move"]) for _, row in data.iterrows()])
    return X, y

def encode_board_state(board):
    """Encode chess board state into numerical array."""
    piece_values = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
    }
    
    state = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        value = piece_values.get(piece.symbol(), 0) if piece else 0
        state.append(value)
    
    # Add turn indicator
    state.append(1 if board.turn else -1)
    return np.array(state)

def decode_prediction(move_str, board):
    """Convert predicted move string back to chess.Move."""
    try:
        return chess.Move.from_uci(move_str)
    except ValueError:
        legal_moves = list(board.legal_moves)
        return legal_moves[0] if legal_moves else None

def generate_training_data(data_path, num_games):
    """Generate enhanced training data."""
    training_data = []
    print(f"[DEBUG] Generating {num_games} games...")
    
    for game_num in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            state = encode_board_state(board)
            move = random.choice(legal_moves)
            training_data.append({
                'board_state': state.tolist(),
                'move': move.uci()
            })
            board.push(move)
    
    df = pd.DataFrame(training_data)
    df.to_csv(data_path, index=False)
    print(f"[DEBUG] Generated {len(training_data)} training positions")
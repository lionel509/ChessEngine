import chess
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pickle
from utils.data_loader import encode_board_state, decode_prediction
from utils.visualization import live_visualize_game

def play_game(model_path):
    # Load the KNN model
    with open(model_path, 'rb') as file:
        knn_model = pickle.load(file)
    
    board = chess.Board()
    board_states = []

    while not board.is_game_over():
        board_states.append(board.copy())
        if board.turn:  # AI's turn (White by default)
            encoded_board = encode_board_state(board).reshape(1, -1)  # Flatten board state for KNN
            predictions = knn_model.predict(encoded_board)
            best_move = decode_prediction(predictions, board)
            board.push(best_move)
        else:  # Opponent's turn (random for now)
            legal_moves = list(board.legal_moves)
            board.push(random.choice(legal_moves))

        live_visualize_game(board_states)

    print("Game Over:", board.result())

if __name__ == "__main__":
    play_game("data/models/knn_model.pkl")

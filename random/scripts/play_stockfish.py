import chess
from stockfish import Stockfish
from sklearn.neighbors import KNeighborsClassifier
import pickle
from utils.data_loader import encode_board_state, decode_prediction
from utils.visualization import live_visualize_game

stockfish_path = "stockfish/stockfish"

stockfish = Stockfish(stockfish_path)

def play_against_stockfish(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    board = chess.Board()
    board_states = []

    stockfish.set_fen_position(board.fen())

    while not board.is_game_over():
        board_states.append(board.copy())
        if board.turn:  # AI's turn (White)
            encoded_board = encode_board_state(board)
            predictions = model.predict(encoded_board)
            best_move = decode_prediction(predictions, board)
            board.push(best_move)
        else:  # Stockfish's turn (Black)
            stockfish.set_fen_position(board.fen())
            stockfish_move = stockfish.get_best_move()
            board.push(chess.Move.from_uci(stockfish_move))

        live_visualize_game(board_states)

    print("Game Over:", board.result())

if __name__ == "__main__":
    play_against_stockfish("data/models/knn_model.pkl")
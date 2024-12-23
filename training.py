import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import chess
import os
from utils.data_loader import encode_board_state, decode_prediction
from  stockfish import Stockfish
from utils.learning_pathways_gui import LearningPathwaysGUI
import pygame
import json
import shutil
from utils.visualization import (
    ChessVisualizer, 
    live_visualize_game, 
    save_training_results
)
import time
import random

def initialize_chess_gui():
    """Initialize chess GUI with error handling."""
    try:
        pygame.init()
        if pygame.display.get_init():
            screen = pygame.display.set_mode((800, 800))
            pygame.display.set_caption("Chess Game: ML vs Stockfish")
            print("[DEBUG] GUI initialized successfully")
            return screen
        else:
            print("[ERROR] Failed to initialize display")
            return None
    except Exception as e:
        print(f"[ERROR] GUI initialization failed: {str(e)}")
        return None

def draw_board(screen, board):
    """Draw chess board with error handling for assets."""
    try:
        if not screen:
            print("[ERROR] No screen surface available")
            return False

        colors = [pygame.Color(235, 236, 208), pygame.Color(119, 149, 86)]
        square_size = 800 // 8

        # Draw board squares
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(screen, color, 
                               pygame.Rect(col * square_size, row * square_size, 
                                         square_size, square_size))

        # Draw pieces
        asset_path = "assets"
        if not os.path.exists(asset_path):
            print(f"[ERROR] Assets directory not found at {asset_path}")
            return False

        for square, piece in board.piece_map().items():
            piece_file = os.path.join(asset_path, f"{piece.symbol()}.png")
            if not os.path.exists(piece_file):
                print(f"[ERROR] Piece asset not found: {piece_file}")
                continue

            try:
                piece_image = pygame.image.load(piece_file)
                piece_image = pygame.transform.scale(piece_image, 
                                                  (square_size, square_size))
                row, col = divmod(square, 8)
                screen.blit(piece_image, (col * square_size, row * square_size))
            except pygame.error as e:
                print(f"[ERROR] Failed to load piece image {piece_file}: {str(e)}")
                continue

        return True

    except Exception as e:
        print(f"[ERROR] Draw board failed: {str(e)}")
        return False

def update_gui(screen, board):
    """Update GUI with error handling."""
    try:
        if screen and draw_board(screen, board):
            pygame.display.flip()
            return True
        return False
    except Exception as e:
        print(f"[ERROR] GUI update failed: {str(e)}")
        return False

def plot_training_results(train_scores, val_scores, neighbors_range):
    """Plot training and validation accuracy over different k values."""
    print("[DEBUG] Plotting training and validation results...")
    # Create a plot to visualize how accuracy changes with different k values
    plt.figure(figsize=(10, 6))
    plt.plot(neighbors_range, train_scores, label="Training Accuracy", marker="o")
    plt.plot(neighbors_range, val_scores, label="Validation Accuracy", marker="o")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy vs. k")
    plt.legend()
    plt.grid()
    plt.show()

def generate_training_data(data_path, num_games=50):
    """Generate chess game data for training."""
    training_data = []
    
    print(f"[DEBUG] Generating {num_games} games...")
    for game_num in range(num_games):
        print(f"[DEBUG] Generating game {game_num + 1}...")
        board = chess.Board()
        
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
                
            # Store current position and chosen move
            state = encode_board_state(board)
            move = random.choice(legal_moves)
            training_data.append({
                'board_state': state.tolist(),
                'move': move.uci()
            })
            
            board.push(move)
    
    # Save to CSV
    df = pd.DataFrame(training_data)
    df.to_csv(data_path, index=False)
    print(f"[DEBUG] Generated {len(training_data)} training positions")

def train_knn_model(data_path, model_path, model, model_name, model_num, total_models):
    """Train single KNN model with progress tracking."""
    try:
        start_time = time.time()
        print(f"\n[DEBUG] Training model {model_num}/{total_models}: {model_name}")
        
        # Load and prep data
        df = pd.read_csv(data_path)
        X = np.array([np.array(eval(state)) for state in df['board_state']])
        y = df['move'].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        # Training with progress updates
        print(f"[DEBUG] Fitting model with {len(X_train)} samples...")
        model.fit(X_train, y_train)
        
        # Evaluate with timing
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        elapsed_time = time.time() - start_time
        eta = elapsed_time * (total_models - model_num)
        
        print(f"[DEBUG] Model {model_num}/{total_models} complete:")
        print(f"├── Train accuracy: {train_score:.4f}")
        print(f"├── Val accuracy: {val_score:.4f}")
        print(f"├── Time taken: {elapsed_time:.1f}s")
        print(f"└── ETA remaining: {eta:.1f}s")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return train_score, val_score
        
    except Exception as e:
        print(f"[ERROR] Model training failed: {str(e)}")
        return 0.0, 0.0

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data",
        "data/models",
        "data/training",
        "stockfish"
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[DEBUG] Created directory: {directory}")

def train_multiple_models(data_path, models_config):
    """Train multiple models in parallel with different configurations."""
    model_results = {}
    best_accuracy = 0
    best_model = None
    
    # Load and prepare data once
    print("[DEBUG] Loading training data...")
    df = pd.read_csv(data_path)
    X = np.array([np.array(eval(state)) for state in df['board_state']])
    y = df['move'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    total_models = len(models_config)
    for model_num, (model_name, config) in enumerate(models_config.items(), start=1):
        print(f"\n[DEBUG] Training {model_name}...")
        model_path = os.path.join("data/models", f"{model_name}.pkl")
        
        # Train model with current configuration
        model = KNeighborsClassifier(**config)
        train_score, val_score = train_knn_model(data_path, model_path, model, model_name, model_num, total_models)
        
        # Store results
        model_results[model_name] = {
            'config': config,
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'model_path': model_path
        }
        
        # Save if best model
        if val_score > best_accuracy:
            best_accuracy = val_score
            best_model = model_name
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    
    # Save training results
    with open('data/model_results.json', 'w') as f:
        json.dump(model_results, f, indent=4)
    
    return model_results, best_model

def continuous_training_with_games(data_path, model_save_path, neighbors_range=range(1, 11)):
    """Continuous training with multiple models."""
    try:
        ensure_directories()
        
        # Configure multiple models
        models_config = {
            'knn_basic': {'n_neighbors': 3},
            'knn_weighted': {'n_neighbors': 5, 'weights': 'distance'},
            'knn_large': {'n_neighbors': 7},
            'knn_manhattan': {'n_neighbors': 3, 'metric': 'manhattan'}
        }
        
        iteration = 1
        while True:
            print(f"\n[DEBUG] Starting iteration {iteration}...")
            current_data_path = os.path.join("data/training", f"training_data_{iteration}.csv")
            
            # Generate or augment training data
            if iteration == 1 or not os.path.exists(current_data_path):
                generate_training_data(current_data_path, num_games=50)
            else:
                generate_training_data(current_data_path, num_games=10)
            
            # Train multiple models
            results, best_model = train_multiple_models(current_data_path, models_config)
            
            # Display results
            print("\n[DEBUG] Training Results:")
            for model_name, result in results.items():
                print(f"{model_name}: Val Accuracy = {result['val_accuracy']:.4f}")
            print(f"\nBest Model: {best_model}")
            
            iteration += 1
            
            if input("\nPress Enter to continue or 'q' to quit: ").lower() == 'q':
                break
                
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")

def play_self_game(model_path):
    """Play a chess game where the AI plays against itself."""
    print("[DEBUG] Loading KNN model for self-play...")
    # Load the trained KNN model from file
    with open(model_path, 'rb') as file:
        knn_model = pickle.load(file)

    board = chess.Board()
    print("[DEBUG] Starting self-play game...")
    while not board.is_game_over():
        live_visualize_game([board])  # Display the current board state live
        if board.turn:  # AI's turn (White)
            encoded_board = encode_board_state(board).reshape(1, -1)  # Encode the board state
            predictions = knn_model.predict(encoded_board)  # Predict the best move
            best_move = decode_prediction(predictions, board)  # Decode the predicted move
            print(f"[DEBUG] White plays: {best_move}")
            board.push(best_move)  # Apply the move to the board
        else:  # AI's turn (Black)
            encoded_board = encode_board_state(board).reshape(1, -1)  # Encode the board state
            predictions = knn_model.predict(encoded_board)  # Predict the best move
            best_move = decode_prediction(predictions, board)  # Decode the predicted move
            print(f"[DEBUG] Black plays: {best_move}")
            board.push(best_move)  # Apply the move to the board

    print("[DEBUG] Self-Play Game Over:", board.result())

def play_stockfish_game(model_path):
    try:
        stockfish_path = "stockfish/stockfish-windows-x86-64-avx2.exe"
        stockfish = Stockfish(path=stockfish_path)
        
        with open(model_path, 'rb') as file:
            knn_model = pickle.load(file)
            
        board = chess.Board()
        while not board.is_game_over():
            if board.turn:  # AI's turn
                state = encode_board_state(board).reshape(1, -1)
                move = decode_prediction(knn_model.predict(state)[0], board)
                board.push(move)
            else:  # Stockfish's turn
                stockfish.set_position([str(move) for move in board.move_stack])
                stockfish_move = chess.Move.from_uci(stockfish.get_best_move())
                board.push(stockfish_move)
                
            live_visualize_game(board)
            time.sleep(1)
            
    except Exception as e:
        print(f"[ERROR] Game failed: {str(e)}")

def load_training_history():
    """Load training history from a JSON file."""
    history_path = 'data/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as file:
            return json.load(file)
    else:
        return {'iterations': [], 'accuracies': {}}

def continuous_training_loop():
    """Enhanced training loop with configurable options."""
    try:
        while True:
            print("\nTraining Options:")
            print("1. Quick training (50 games)")
            print("2. Deep training (200 games)")
            print("3. Extended training (500 games)") 
            print("4. View progress")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ")
            
            if choice == '5':
                break
                
            if choice == '4':
                plot_training_progress(load_training_history())
                continue
                
            num_games = {
                '1': 50,
                '2': 200,
                '3': 500
            }.get(choice, 50)
            
            # Enhanced model configurations
            models_config = {
                'knn_weighted_3': {'n_neighbors': 3, 'weights': 'distance', 'metric': 'manhattan'},
                'knn_weighted_5': {'n_neighbors': 5, 'weights': 'distance', 'metric': 'manhattan'},
                'knn_weighted_7': {'n_neighbors': 7, 'weights': 'distance', 'metric': 'manhattan'},
                'knn_euclidean': {'n_neighbors': 5, 'weights': 'distance', 'metric': 'euclidean'}
            }
            
            train_iteration(num_games, models_config)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")

def train_iteration(num_games, models_config):
    """Single training iteration with enhanced features."""
    history = load_training_history()
    iteration = len(history['iterations']) + 1
    
    print(f"\n[DEBUG] Starting training iteration {iteration} with {num_games} games")
    
    data_path = f"data/training/training_data_{iteration}.csv"
    generate_training_data(data_path, num_games)
    
    results, best_model = train_multiple_models(data_path, models_config)
    save_training_results(history, iteration, results, best_model)
    
    return results, best_model

def plot_training_progress(history):
    """Plot training progress over iterations."""
    plt.figure(figsize=(10, 6))
    for model_name, accuracies in history['accuracies'].items():
        plt.plot(history['iterations'], accuracies, label=model_name, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.close()

def encode_enhanced_board_state(board):
    """Enhanced board state encoding with additional features."""
    state = []
    
    # Basic piece positions and values
    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
                   'p':-1, 'n':-3, 'b':-3, 'r':-5, 'q':-9, 'k': 0}
    
    # Material balance
    material_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.symbol(), 0)
            material_score += value
            state.append(value)
        else:
            state.append(0)
            
    # Mobility features
    white_moves = len(list(board.legal_moves))
    board.turn = chess.BLACK
    black_moves = len(list(board.legal_moves))
    board.turn = chess.WHITE
    
    # Add enhanced features
    state.extend([
        material_score,  # Material balance
        white_moves,     # White mobility
        black_moves,     # Black mobility
        int(board.is_check()),  # Check state
        len(board.piece_map()), # Piece count
        int(board.turn),        # Turn indicator
    ])
    
    return np.array(state)

def create_iteration_directory(iteration):
    """Create directory structure for training iteration."""
    base_dir = f"iterations/iteration_{iteration}"
    subdirs = ['models', 'data', 'results', 'plots']
    
    for subdir in [base_dir] + [f"{base_dir}/{d}" for d in subdirs]:
        if not os.path.exists(subdir):
            os.makedirs(subdir)
            print(f"[DEBUG] Created directory: {subdir}")
    
    return {
        'base': base_dir,
        'models': f"{base_dir}/models",
        'data': f"{base_dir}/data",
        'results': f"{base_dir}/results",
        'plots': f"{base_dir}/plots"
    }

def generate_training_data(iteration_paths, num_games=50):
    """Generate chess game data for training with iteration tracking."""
    training_data = []
    data_path = f"{iteration_paths['data']}/training_data.csv"
    
    print(f"[DEBUG] Generating {num_games} games...")
    for game_num in range(num_games):
        print(f"[DEBUG] Generating game {game_num + 1}...")
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
    
    # Save metadata
    metadata = {
        'num_games': num_games,
        'positions': len(training_data),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(f"{iteration_paths['results']}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"[DEBUG] Generated {len(training_data)} training positions")
    return data_path

def train_iteration(iteration, num_games, models_config):
    """Run training iteration with progress tracking."""
    iteration_paths = create_iteration_directory(iteration)
    data_path = generate_training_data(iteration_paths, num_games)
    
    results = {}
    total_models = len(models_config)
    
    print(f"\n[DEBUG] Starting training of {total_models} models...")
    start_time = time.time()
    
    for idx, (model_name, config) in enumerate(models_config.items(), 1):
        model_path = f"{iteration_paths['models']}/{model_name}.pkl"
        knn = KNeighborsClassifier(**config)
        
        train_score, val_score = train_knn_model(
            data_path, model_path, knn, model_name, idx, total_models
        )
        
        results[model_name] = {
            'config': config,
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'model_path': model_path
        }
    
    total_time = time.time() - start_time
    print(f"\n[DEBUG] Training complete in {total_time:.1f}s")
    
    save_training_results(iteration_paths, results)
    return results

def plot_training_metrics(results, plot_path):
    """Plot training metrics and save to file."""
    plt.figure(figsize=(10, 6))
    for model_name, result in results.items():
        plt.plot(result['config']['n_neighbors'], result['val_accuracy'], label=model_name, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_path}/training_metrics.png")
    plt.close()

def view_all_results():
    """View all training results."""
    results_path = 'iterations'
    for iteration_dir in os.listdir(results_path):
        iteration_path = os.path.join(results_path, iteration_dir)
        if os.path.isdir(iteration_path):
            results_file = os.path.join(iteration_path, 'results', 'results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    print(f"\nResults for {iteration_dir}:")
                    for model_name, result in results.items():
                        print(f"{model_name}: Val Accuracy = {result['val_accuracy']:.4f}")

def continuous_training_loop():
    """Main training loop with organized iterations."""
    iteration = 1
    models_config = {
        'knn_weighted_3': {'n_neighbors': 3, 'weights': 'distance', 'metric': 'manhattan'},
        'knn_weighted_5': {'n_neighbors': 5, 'weights': 'distance', 'metric': 'manhattan'},
        'knn_weighted_7': {'n_neighbors': 7, 'weights': 'distance', 'metric': 'manhattan'},
        'knn_euclidean': {'n_neighbors': 5, 'weights': 'distance', 'metric': 'euclidean'}
    }
    while True:
        print(f"\n=== Training Iteration {iteration} ===")
        print("1. Quick training (50 games)")
        print("2. Deep training (200 games)")
        print("3. View results")
        print("4. Exit")
        
        choice = input("\nSelect option: ")
        if choice == '4':
            break
            
        if choice == '3':
            view_all_results()
            continue
            
        num_games = 200 if choice == '2' else 50
        train_iteration(iteration, num_games, models_config)
        
        iteration += 1

# Specify file paths and parameters
data_path = "data/training_games.csv"
model_save_path = "data/models/knn_model.pkl"

# Continuous Training with Self-Play and Stockfish Games
continuous_training_with_games(data_path, model_save_path)

if __name__ == "__main__":
    continuous_training_loop()

class TrainingProgress:
    def __init__(self, total_models, total_samples):
        self.start_time = time.time()
        self.total_models = total_models
        self.total_samples = total_samples
        self.current_model = 0
        self.processed_samples = 0
        self.model_times = []

    def update_model_progress(self, model_num, samples_processed):
        """Update progress for current model."""
        self.current_model = model_num
        self.processed_samples = samples_processed
        current_time = time.time()
        elapsed = current_time - self.start_time

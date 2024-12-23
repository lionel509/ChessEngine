import chess
import chess.svg
import os
import time
import pygame
import json
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, SVG

class ChessVisualizer:
    def __init__(self, output_mode='console'):
        self.output_mode = output_mode
        self.history = []
        self.move_count = 0
        if output_mode == 'gui':
            pygame.init()
            self.screen = pygame.display.set_mode((800, 800))
            self.square_size = 800 // 8
            self.load_piece_images()

    def load_piece_images(self):
        self.piece_images = {}
        pieces = 'rnbqkpRNBQKP'
        for piece in pieces:
            self.piece_images[piece] = pygame.image.load(f"assets/{piece}.png")
            self.piece_images[piece] = pygame.transform.scale(
                self.piece_images[piece], 
                (self.square_size, self.square_size)
            )

    def display_board(self, board, clear=True):
        if self.output_mode == 'console':
            if clear:
                os.system('cls' if os.name == 'nt' else 'clear')
            print("\n" + str(board))
        elif self.output_mode == 'svg':
            svg = chess.svg.board(board=board)
            display(SVG(svg))
        elif self.output_mode == 'gui':
            self.draw_board(board)
        
    def update_progress(self, current, total):
        progress = int(50 * current / total)
        print(f"\rProgress: [{'=' * progress}{' ' * (50-progress)}] {current}/{total}", end='')

    def live_visualize_game(self, board, move=None, player=None, clear=True):
        """Live visualization of chess game with move history."""
        self.move_count += 1
        self.history.append(board.fen())
        
        if clear:
            os.system('cls' if os.name == 'nt' else 'clear')
            
        print(f"\nMove {self.move_count}:")
        if move:
            print(f"{'White' if player else 'Black'} plays: {move}")
        
        self.display_board(board, clear=False)
        print("\nMove history:")
        for i, move in enumerate(board.move_stack[-5:], 1):
            print(f"{i}. {move}")
        
        time.sleep(0.5)  # Short delay for visualization

def save_training_results(iteration_paths, results):
    """Save training results and metrics."""
    try:
        # Save JSON results
        results_file = os.path.join(iteration_paths['results'], 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        # Create plots directory if needed
        plots_dir = iteration_paths['plots']
        os.makedirs(plots_dir, exist_ok=True)

        # Generate performance plot
        create_performance_plot(results, plots_dir)
        
        print(f"[DEBUG] Saved results to {results_file}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to save results: {str(e)}")
        return False

def create_performance_plot(results, plot_dir):
    """Create and save performance visualization."""
    plt.figure(figsize=(12, 6))
    
    models = list(results.keys())
    train_scores = [results[m]['train_accuracy'] for m in models]
    val_scores = [results[m]['val_accuracy'] for m in models]
    
    x = range(len(models))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], train_scores, width, 
            label='Training', color='skyblue')
    plt.bar([i + width/2 for i in x], val_scores, width,
            label='Validation', color='lightgreen')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_dir, 'performance.png'))
    plt.close()

def live_visualize_game(board, move=None, player=None):
    """Live game visualization."""
    visualizer = ChessVisualizer()
    visualizer.live_visualize_game(board, move, player)
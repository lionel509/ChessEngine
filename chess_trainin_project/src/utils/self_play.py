import chess
import chess.engine
import chess.pgn
from datetime import datetime
import time
from pathlib import Path
from tqdm import tqdm

class SelfPlayGenerator:
    def __init__(self, stockfish_path: str = None):
        self.stockfish_path = stockfish_path
        if stockfish_path:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
    def generate_games(self, num_games: int, output_path: str, time_limit: float = 0.1):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        with open(output_path, 'w') as pgn_file:
            # Add progress bar
            pbar = tqdm(total=num_games, desc="Generating games")
            
            for game_id in range(num_games):
                board = chess.Board()
                game = chess.pgn.Game()
                
                # Set game metadata
                game.headers["Event"] = f"Training Game {game_id+1}"
                game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
                game.headers["Round"] = str(game_id + 1)
                
                move_count = 0
                while not board.is_game_over() and move_count < 100:  # Limit moves per game
                    if self.stockfish_path:
                        result = self.engine.play(board, chess.engine.Limit(time=time_limit))
                        move = result.move
                    else:
                        moves = list(board.legal_moves)
                        move = moves[hash(str(board)) % len(moves)]
                    
                    board.push(move)
                    move_count += 1
                
                game.add_line(board.move_stack)
                print(game, file=pgn_file, end='\n\n')
                
                # Update progress
                pbar.update(1)
                if (game_id + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (game_id + 1) / elapsed
                    eta = (num_games - (game_id + 1)) / rate
                    pbar.set_postfix({
                        'ETA': f'{eta:.1f}s',
                        'Rate': f'{rate:.1f} games/s'
                    })
            
            pbar.close()
    
    def close(self):
        if hasattr(self, 'engine'):
            self.engine.quit()
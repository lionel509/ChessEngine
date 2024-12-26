import multiprocessing as mp
from pathlib import Path
from src.utils.self_play import SelfPlayGenerator
import os
import time
from tqdm import tqdm

def generate_game_batch(args):
    batch_id, num_games, stockfish_path = args
    generator = SelfPlayGenerator(stockfish_path)
    output_path = f'data/temp_games_{batch_id}.pgn'
    
    try:
        generator.generate_games(
            num_games=num_games,
            output_path=output_path
        )
    finally:
        generator.close()
    
    return output_path

def merge_pgn_files(temp_files, final_path):
    with open(final_path, 'w') as outfile:
        for temp_file in temp_files:
            with open(temp_file) as infile:
                outfile.write(infile.read())
            os.remove(temp_file)

def main():
    # Setup
    stockfish_path = r"C:\Users\Lionel\Downloads\ChessEngine\stockfish\stockfish-windows-x86-64-avx2.exe"
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    total_games = 1000
    games_per_process = total_games // num_processes
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    # Prepare batch arguments
    batch_args = [
        (i, games_per_process, stockfish_path)
        for i in range(num_processes)
    ]
    
    print(f"Starting parallel generation using {num_processes} processes")
    start_time = time.time()
    
    # Run parallel generation
    with mp.Pool(num_processes) as pool:
        temp_files = list(tqdm(
            pool.imap(generate_game_batch, batch_args),
            total=len(batch_args),
            desc="Generating game batches"
        ))
    
    # Merge results
    print("Merging generated games...")
    merge_pgn_files(temp_files, 'data/games.pgn')
    
    total_time = time.time() - start_time
    print(f"Generated {total_games} games in {total_time:.1f} seconds")

if __name__ == '__main__':
    main()
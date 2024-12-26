import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from src.models.knn_model import KNNChessModel  
from src.utils.data_loader import create_data_loaders
import multiprocessing as mp
from tqdm import tqdm
import logging
import time
import psutil
from pathlib import Path

def setup_logging():
    logger = logging.getLogger('train_knn')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler('train_knn_debug.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def train_model_chunk(args):
    chunk_id, X_chunk, y_chunk, n_neighbors = args
    model = KNNChessModel(n_neighbors=n_neighbors)
    metrics = model.train(X_chunk, y_chunk)
    return chunk_id, model, metrics

def train_knn_model():
    logger = setup_logging()
    start_time = time.time()
    
    try:
        logger.debug(f"Starting training process. Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        
        # Load data
        logger.info("Loading chess games...")
        pgn_files = ['data/games.pgn']  # Add your PGN files
        logger.debug(f"Loading from files: {pgn_files}")
        
        train_loader, val_loader = create_data_loaders(pgn_files, batch_size=1000)
        logger.debug(f"Data loaders created. Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        
        # Convert data loaders to numpy arrays with progress bar
        X_train = []
        y_train = []
        
        with tqdm(train_loader, desc="Loading training data") as pbar:
            for positions, moves in pbar:
                X_train.append(positions.numpy())
                y_train.append(moves.numpy())
                batch_size = len(positions)
                logger.debug(f"Loaded batch of size {batch_size}")
                pbar.set_postfix({'batch_size': batch_size})
        
        logger.debug("Concatenating arrays...")
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        logger.debug(f"Final data shapes: X={X_train.shape}, y={y_train.shape}")
        
        # Split data into chunks for parallel training
        n_chunks = mp.cpu_count() - 1
        chunk_size = len(X_train) // n_chunks
        
        # Prepare chunks with different n_neighbors values
        train_args = []
        neighbors_range = [3, 5, 7, 9, 11]  # Different k values
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_chunks - 1 else len(X_train)
            n_neighbors = neighbors_range[i % len(neighbors_range)]
            train_args.append((
                i,
                X_train[start_idx:end_idx],
                y_train[start_idx:end_idx],
                n_neighbors
            ))
        
        # Train models in parallel
        logger.info(f"Training {n_chunks} models in parallel...")
        with mp.Pool(n_chunks) as pool:
            results = list(tqdm(
                pool.imap(train_model_chunk, train_args),
                total=len(train_args),
                desc="Training models"
            ))
        
        # Save ensemble
        logger.info("Saving ensemble models...")
        Path('models').mkdir(exist_ok=True)
        for chunk_id, model, metrics in results:
            joblib.dump(model, f'models/knn_model_chunk_{chunk_id}.joblib')
            logger.info(f"Chunk {chunk_id} metrics: {metrics}")
        
        # Save ensemble metadata
        ensemble_meta = {
            'n_models': n_chunks,
            'model_files': [f'knn_model_chunk_{i}.joblib' for i in range(n_chunks)]
        }
        joblib.dump(ensemble_meta, 'models/ensemble_meta.joblib')
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    train_knn_model()
import torch
from torch.utils.data import Dataset
import chess.pgn
import chess
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

class ChessDataset(Dataset):
    def __init__(self, pgn_files: List[str], max_positions: int = None):
        self.positions = []
        self.moves = []
        self.load_pgn_files(pgn_files, max_positions)
        
    def load_pgn_files(self, pgn_files: List[str], max_positions: int = None):
        """Load chess positions and moves from PGN files"""
        position_count = 0
        
        for pgn_path in pgn_files:
            with open(pgn_path) as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                        
                    board = game.board()
                    for move in game.mainline_moves():
                        self.positions.append(self.board_to_tensor(board))
                        self.moves.append(self.move_to_index(move))
                        board.push(move)
                        
                        position_count += 1
                        if max_positions and position_count >= max_positions:
                            return
                            
    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """Convert chess board to tensor representation"""
        planes = np.zeros((12, 8, 8), dtype=np.float32)
        
        # Piece placement
        piece_idx = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                planes[piece_idx[piece.symbol()]][rank][file] = 1
                
        return torch.FloatTensor(planes)
    
    def move_to_index(self, move: chess.Move) -> int:
        """Convert chess move to index"""
        return move.from_square * 64 + move.to_square
    
    def index_to_move(self, index: int) -> chess.Move:
        """Convert index to chess move"""
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.positions[idx], self.moves[idx]
    
    def augment_position(self, position: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to position"""
        # Horizontal flip
        if np.random.random() > 0.5:
            position = torch.flip(position, [2])  # flip horizontally
        return position

def create_data_loaders(
    pgn_files: List[str],
    batch_size: int = 32,
    train_split: float = 0.8,
    max_positions: int = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation data loaders"""
    dataset = ChessDataset(pgn_files, max_positions)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader
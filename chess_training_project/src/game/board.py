import numpy as np
from typing import Tuple, List, Optional

class ChessBoard:
    def __init__(self):
        # Initialize 8x8 board
        self.board = np.zeros((8, 8), dtype=int)
        self.piece_map = {
            'p': 1,  # white pawn
            'P': -1, # black pawn
            'r': 2,  # white rook
            'R': -2, # black rook
            'n': 3,  # white knight
            'N': -3, # black knight
            'b': 4,  # white bishop
            'B': -4, # black bishop
            'q': 5,  # white queen
            'Q': -5, # black queen
            'k': 6,  # white king
            'K': -6  # black king
        }
        self.reset_board()

    def reset_board(self):
        """Reset the board to starting position"""
        # Initialize pawns
        self.board[1, :] = self.piece_map['p']  # white pawns
        self.board[6, :] = self.piece_map['P']  # black pawns
        
        # Initialize other pieces
        back_row = [
            self.piece_map['r'], self.piece_map['n'], self.piece_map['b'],
            self.piece_map['q'], self.piece_map['k'], self.piece_map['b'],
            self.piece_map['n'], self.piece_map['r']
        ]
        self.board[0, :] = back_row  # white pieces
        self.board[7, :] = [-x for x in back_row]  # black pieces

    def is_valid_move(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if move is valid"""
        if not (0 <= start[0] <= 7 and 0 <= start[1] <= 7 and 
                0 <= end[0] <= 7 and 0 <= end[1] <= 7):
            return False
            
        piece = self.board[start[0]][start[1]]
        if piece == 0:  # Empty square
            return False
            
        # Add piece-specific move validation here
        return self._is_path_clear(start, end)

    def _is_path_clear(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if path between start and end is clear"""
        x1, y1 = start
        x2, y2 = end
        
        # Calculate step direction
        dx = 0 if x2 == x1 else (x2 - x1) // abs(x2 - x1)
        dy = 0 if y2 == y1 else (y2 - y1) // abs(y2 - y1)
        
        x, y = x1 + dx, y1 + dy
        while (x, y) != end:
            if self.board[x][y] != 0:
                return False
            x, y = x + dx, y + dy
            
        return True

    def make_move(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Make a move on the board"""
        if not self.is_valid_move(start, end):
            return False
            
        self.board[end[0]][end[1]] = self.board[start[0]][start[1]]
        self.board[start[0]][start[1]] = 0
        return True

    def get_legal_moves(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all legal moves for piece at position"""
        legal_moves = []
        for i in range(8):
            for j in range(8):
                if self.is_valid_move(position, (i, j)):
                    legal_moves.append((i, j))
        return legal_moves

    def is_checkmate(self) -> bool:
        """Check if current position is checkmate"""
        # Implement checkmate detection
        return False

    def get_board_state(self) -> np.ndarray:
        """Return current board state"""
        return self.board.copy()

    def __str__(self) -> str:
        """String representation of the board"""
        symbols = {
            1: '♙', -1: '♟',
            2: '♖', -2: '♜',
            3: '♘', -3: '♞',
            4: '♗', -4: '♝',
            5: '♕', -5: '♛',
            6: '♔', -6: '♚',
            0: '·'
        }
        
        board_str = ''
        for i in range(7, -1, -1):
            board_str += f'{i+1} '
            for j in range(8):
                board_str += f'{symbols[self.board[i][j]]} '
            board_str += '\n'
        board_str += '  a b c d e f g h'
        return board_str
from typing import List, Tuple, Dict
import numpy as np

class ChessRules:
    def __init__(self):
        self.directions = {
            'pawn': {'white': [(1, 0)], 'black': [(-1, 0)]},
            'rook': [(0, 1), (0, -1), (1, 0), (-1, 0)],
            'knight': [(2, 1), (2, -1), (-2, 1), (-2, -1),
                      (1, 2), (1, -2), (-1, 2), (-1, -2)],
            'bishop': [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            'queen': [(0, 1), (0, -1), (1, 0), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)],
            'king': [(0, 1), (0, -1), (1, 0), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)]
        }
        self.piece_values = {
            'pawn': 1,
            'knight': 3,
            'bishop': 3,
            'rook': 5,
            'queen': 9,
            'king': 0
        }

    def get_piece_moves(self, piece: str, position: Tuple[int, int], board: np.ndarray, color: str) -> List[Tuple[int, int]]:
        """Get all possible moves for a piece"""
        moves = []
        if piece.lower() == 'pawn':
            moves = self._get_pawn_moves(position, board, color)
        elif piece.lower() in ['rook', 'bishop', 'queen']:
            moves = self._get_sliding_moves(position, board, self.directions[piece.lower()], color)
        elif piece.lower() == 'knight':
            moves = self._get_knight_moves(position, board, color)
        elif piece.lower() == 'king':
            moves = self._get_king_moves(position, board, color)
        return moves

    def _get_pawn_moves(self, pos: Tuple[int, int], board: np.ndarray, color: str) -> List[Tuple[int, int]]:
        moves = []
        direction = 1 if color == 'white' else -1
        x, y = pos

        # Forward move
        if 0 <= x + direction < 8 and board[x + direction][y] == 0:
            moves.append((x + direction, y))
            # Initial two-square move
            if (color == 'white' and x == 1) or (color == 'black' and x == 6):
                if board[x + 2*direction][y] == 0:
                    moves.append((x + 2*direction, y))

        # Captures
        for dy in [-1, 1]:
            if 0 <= x + direction < 8 and 0 <= y + dy < 8:
                target = board[x + direction][y + dy]
                if (color == 'white' and target < 0) or (color == 'black' and target > 0):
                    moves.append((x + direction, y + dy))

        return moves

    def _get_sliding_moves(self, pos: Tuple[int, int], board: np.ndarray, directions: List[Tuple[int, int]], color: str) -> List[Tuple[int, int]]:
        moves = []
        x, y = pos
        
        for dx, dy in directions:
            curr_x, curr_y = x + dx, y + dy
            while 0 <= curr_x < 8 and 0 <= curr_y < 8:
                target = board[curr_x][curr_y]
                if target == 0:
                    moves.append((curr_x, curr_y))
                elif (color == 'white' and target < 0) or (color == 'black' and target > 0):
                    moves.append((curr_x, curr_y))
                    break
                else:
                    break
                curr_x, curr_y = curr_x + dx, curr_y + dy
                
        return moves

    def _get_knight_moves(self, pos: Tuple[int, int], board: np.ndarray, color: str) -> List[Tuple[int, int]]:
        moves = []
        x, y = pos
        
        for dx, dy in self.directions['knight']:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                target = board[new_x][new_y]
                if target == 0 or (color == 'white' and target < 0) or (color == 'black' and target > 0):
                    moves.append((new_x, new_y))
                    
        return moves

    def _get_king_moves(self, pos: Tuple[int, int], board: np.ndarray, color: str) -> List[Tuple[int, int]]:
        moves = []
        x, y = pos
        
        for dx, dy in self.directions['king']:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                target = board[new_x][new_y]
                if target == 0 or (color == 'white' and target < 0) or (color == 'black' and target > 0):
                    moves.append((new_x, new_y))
                    
        return moves

    def is_check(self, board: np.ndarray, color: str) -> bool:
        """Determine if the king is in check"""
        # Find king position
        king_value = 6 if color == 'white' else -6
        king_pos = tuple(p[0] for p in np.where(board == king_value))
        
        # Check if any opponent piece can capture the king
        for i in range(8):
            for j in range(8):
                piece = board[i][j]
                if (color == 'white' and piece < 0) or (color == 'black' and piece > 0):
                    if king_pos in self.get_piece_moves(str(abs(piece)), (i, j), board, 'black' if color == 'white' else 'white'):
                        return True
        return False

    def is_checkmate(self, board: np.ndarray, color: str) -> bool:
        """Determine if the position is checkmate"""
        if not self.is_check(board, color):
            return False
            
        # Check if any move can get out of check
        for i in range(8):
            for j in range(8):
                piece = board[i][j]
                if (color == 'white' and piece > 0) or (color == 'black' and piece < 0):
                    moves = self.get_piece_moves(str(abs(piece)), (i, j), board, color)
                    for move in moves:
                        # Try move
                        test_board = board.copy()
                        test_board[move[0]][move[1]] = test_board[i][j]
                        test_board[i][j] = 0
                        if not self.is_check(test_board, color):
                            return False
        return True
import chess
import numpy as np
from typing import List, Tuple, Dict, Optional
import re

class ChessPreprocessor:
    def __init__(self):
        self.piece_values = {
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
            'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
        }
        
    def validate_fen(self, fen: str) -> bool:
        """Validate FEN string format"""
        try:
            chess.Board(fen)
            return True
        except ValueError:
            return False
            
    def extract_features(self, board: chess.Board) -> Dict:
        """Extract numerical features from position"""
        features = {
            'material_balance': self._calculate_material(board),
            'piece_mobility': self._calculate_mobility(board),
            'king_safety': self._evaluate_king_safety(board),
            'pawn_structure': self._evaluate_pawn_structure(board),
            'center_control': self._evaluate_center_control(board)
        }
        return features
        
    def _calculate_material(self, board: chess.Board) -> int:
        """Calculate material balance"""
        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                material += self.piece_values[piece.symbol()]
        return material
        
    def _calculate_mobility(self, board: chess.Board) -> Tuple[int, int]:
        """Calculate piece mobility for both sides"""
        white_mobility = len(list(board.legal_moves))
        board.turn = not board.turn
        black_mobility = len(list(board.legal_moves))
        board.turn = not board.turn
        return white_mobility, black_mobility
        
    def _evaluate_king_safety(self, board: chess.Board) -> Dict:
        """Evaluate king safety for both sides"""
        safety = {'white': 0, 'black': 0}
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square:
                # Check pawn shield
                safety['white' if color else 'black'] = self._count_king_defenders(board, king_square)
                
        return safety
        
    def _count_king_defenders(self, board: chess.Board, king_square: int) -> int:
        """Count pieces defending the king"""
        defenders = 0
        for square in board.attacks(king_square):
            piece = board.piece_at(square)
            if piece and piece.color == board.piece_at(king_square).color:
                defenders += 1
        return defenders
        
    def _evaluate_pawn_structure(self, board: chess.Board) -> Dict:
        """Evaluate pawn structure"""
        structure = {
            'doubled_pawns': self._count_doubled_pawns(board),
            'isolated_pawns': self._count_isolated_pawns(board),
            'passed_pawns': self._count_passed_pawns(board)
        }
        return structure
        
    def _count_doubled_pawns(self, board: chess.Board) -> Dict:
        """Count doubled pawns for both sides"""
        doubled = {'white': 0, 'black': 0}
        for file in range(8):
            white_pawns = 0
            black_pawns = 0
            for rank in range(8):
                square = rank * 8 + file
                piece = board.piece_at(square)
                if piece:
                    if piece.symbol() == 'P':
                        white_pawns += 1
                    elif piece.symbol() == 'p':
                        black_pawns += 1
            doubled['white'] += max(0, white_pawns - 1)
            doubled['black'] += max(0, black_pawns - 1)
        return doubled
        
    def _count_isolated_pawns(self, board: chess.Board) -> Dict:
        """Count isolated pawns"""
        isolated = {'white': 0, 'black': 0}
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if self._is_isolated_pawn(board, square):
                    isolated['white' if piece.color else 'black'] += 1
        return isolated
        
    def _is_isolated_pawn(self, board: chess.Board, square: int) -> bool:
        """Check if pawn is isolated"""
        file = square % 8
        color = board.piece_at(square).color
        adjacent_files = []
        if file > 0:
            adjacent_files.append(file - 1)
        if file < 7:
            adjacent_files.append(file + 1)
            
        for adj_file in adjacent_files:
            for rank in range(8):
                adj_square = rank * 8 + adj_file
                piece = board.piece_at(adj_square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    return False
        return True
        
    def _count_passed_pawns(self, board: chess.Board) -> Dict:
        """Count passed pawns"""
        passed = {'white': 0, 'black': 0}
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if self._is_passed_pawn(board, square):
                    passed['white' if piece.color else 'black'] += 1
        return passed
        
    def _is_passed_pawn(self, board: chess.Board, square: int) -> bool:
        """Check if pawn is passed"""
        piece = board.piece_at(square)
        file = square % 8
        rank = square // 8
        
        if piece.color:  # White pawn
            target_ranks = range(rank + 1, 8)
        else:  # Black pawn
            target_ranks = range(rank - 1, -1, -1)
            
        for r in target_ranks:
            for f in [file - 1, file, file + 1]:
                if 0 <= f <= 7:
                    check_square = r * 8 + f
                    blocking_piece = board.piece_at(check_square)
                    if blocking_piece and blocking_piece.piece_type == chess.PAWN and blocking_piece.color != piece.color:
                        return False
        return True
        
    def _evaluate_center_control(self, board: chess.Board) -> Dict:
        """Evaluate center square control"""
        center_squares = [27, 28, 35, 36]  # e4, d4, e5, d5
        control = {'white': 0, 'black': 0}
        
        for square in center_squares:
            white_attackers = len(list(board.attackers(chess.WHITE, square)))
            black_attackers = len(list(board.attackers(chess.BLACK, square)))
            control['white'] += white_attackers
            control['black'] += black_attackers
            
        return control
import tkinter as tk
from tkinter import ttk
import chess
import logging
import numpy as np
from typing import Dict, Any

class ChessGUI:
    def __init__(self, models):
        self.models = models
        self.board = chess.Board()
        self.selected_square = None
        self.buttons = {}
        
        # Add piece symbols mapping
        self.piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        self.root = tk.Tk()
        self.root.title("Chess Engine")
        
        # Create main frames
        top_frame = ttk.Frame(self.root)
        top_frame.pack(pady=5)
        
        # Model selection
        ttk.Label(top_frame, text="Select Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, values=list(self.models.keys()))
        model_combo.pack(side=tk.LEFT, padx=5)
        
        # Create main content frame
        content_frame = ttk.Frame(self.root)
        content_frame.pack(expand=True, fill=tk.BOTH)
        
        # Board frame on left
        board_frame = ttk.Frame(content_frame)
        board_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Move history frame on right
        history_frame = ttk.Frame(content_frame)
        history_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH)
        
        # Move history display
        ttk.Label(history_frame, text="Move History").pack()
        self.move_history = tk.Text(history_frame, width=30, height=20)
        self.move_history.pack()

        # Create chess board
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7-rank)
                color = '#FFFFFF' if (rank + file) % 2 == 0 else '#808080'
                btn = tk.Button(board_frame, width=3, height=1, bg=color,
                              command=lambda s=square: self.on_square_click(s))
                btn.grid(row=rank, column=file)
                self.buttons[square] = btn
                
        # Status bar
        self.status_var = tk.StringVar(value="Select a model to begin")
        ttk.Label(self.root, textvariable=self.status_var).pack(pady=5)
        
        self.update_board()
        self.logger = logging.getLogger(__name__)
        
    def update_board(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            text = self.piece_symbols.get(piece.symbol(), ' ') if piece else ' '
            self.buttons[square].config(text=text)
            
    def on_square_click(self, square):
        if not self.model_var.get():
            self.status_var.set("Please select a model first")
            return
            
        try:
            if self.selected_square is None:
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
                    self.buttons[square].config(bg='yellow')
            else:
                move = chess.Move(self.selected_square, square)
                # Clear selection
                self.buttons[self.selected_square].config(
                    bg='#FFFFFF' if (self.selected_square + (self.selected_square // 8)) % 2 == 0 else '#808080'
                )
                self.selected_square = None
                
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.update_board()
                    self.update_move_history(move)
                    self.make_ai_move()
                else:
                    self.status_var.set("Illegal move")
        except Exception as e:
            self.logger.error(f"Error in move handling: {e}")
            self.status_var.set("Error processing move")

    def make_ai_move(self):
        try:
            if self.board.is_game_over():
                self.status_var.set(f"Game Over: {self.board.outcome().result()}")
                return
                
            model = self.models[self.model_var.get()]
            board_tensor = self.board_to_tensor()
            ai_move = model.predict(board_tensor)
            
            if isinstance(ai_move, tuple):
                ai_move = chess.Move(ai_move[0], ai_move[1])
                
            if ai_move in self.board.legal_moves:
                self.board.push(ai_move)
                self.update_board()
                self.update_move_history(ai_move)
                
                if self.board.is_game_over():
                    self.status_var.set(f"Game Over: {self.board.outcome().result()}")
            else:
                self.status_var.set("AI made illegal move")
        except Exception as e:
            self.logger.error(f"Error in AI move: {e}")
            self.status_var.set("Error making AI move")
            
    def update_move_history(self, move):
        try:
            if move in self.board.legal_moves or move in self.board.move_stack:
                move_text = move.uci()
                move_number = (len(self.board.move_stack) + 1) // 2
                
                if self.board.turn:  # Black to move
                    self.move_history.insert(tk.END, f"{move_number}. ... {move_text}\n")
                else:  # White to move
                    self.move_history.insert(tk.END, f"{move_number}. {move_text} ")
                
                self.move_history.see(tk.END)
        except Exception as e:
            self.logger.error(f"Error updating move history: {e}")

    def board_to_tensor(self) -> np.ndarray:
        """Convert current board state to tensor representation"""
        # Initialize 12 planes (6 piece types x 2 colors)
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        # Piece type mapping
        piece_idx = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
        }
        
        try:
            # Fill tensor with piece positions
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece:
                    rank, file = divmod(square, 8)
                    tensor[piece_idx[piece.symbol()]][rank][file] = 1.0
                    
            return tensor
            
        except Exception as e:
            self.logger.error(f"Error converting board to tensor: {e}")
            raise

    def run(self):
        self.root.mainloop()
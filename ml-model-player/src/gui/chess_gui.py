import tkinter as tk
from tkinter import ttk
import chess

class ChessGUI:
    def __init__(self, models):
        self.root = tk.Tk()
        self.root.title("Chess AI Player")
        self.models = models
        self.board = chess.Board()
        self.buttons = {}
        self.selected_square = None
        self.piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        self.setup_gui()
        
    def setup_gui(self):
        # Top frame for controls
        top_frame = ttk.Frame(self.root)
        top_frame.pack(pady=5)
        
        ttk.Label(top_frame, text="Select Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, values=list(self.models.keys()))
        model_combo.pack(side=tk.LEFT, padx=5)
        
        # Board frame
        board_frame = ttk.Frame(self.root)
        board_frame.pack(padx=10, pady=10)
        
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
        
    def update_board(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            text = self.piece_symbols.get(piece.symbol(), ' ') if piece else ' '
            self.buttons[square].config(text=text)
            
    def on_square_click(self, square):
        if not self.model_var.get():
            self.status_var.set("Please select a model first")
            return
            
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.buttons[square].config(bg='yellow')
        else:
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.update_board()
                self.make_ai_move()
            
            # Reset selection
            self.buttons[self.selected_square].config(
                bg='#FFFFFF' if (chess.square_rank(self.selected_square) + 
                chess.square_file(self.selected_square)) % 2 == 0 else '#808080'
            )
            self.selected_square = None
            
    def make_ai_move(self):
        if self.board.is_game_over():
            self.status_var.set(f"Game Over! Result: {self.board.result()}")
            return
            
        model = self.models[self.model_var.get()]
        # TODO: Implement AI move selection based on your model
        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            self.board.push(legal_moves[0])
            self.update_board()
            
    def run(self):
        self.root.mainloop()
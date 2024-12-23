import tkinter as tk
from tkinter import Canvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import chess
import chess.svg
from PIL import Image, ImageTk
from io import BytesIO

def update_chess_board(canvas, board):
    """Update the chessboard display in the GUI without using cairosvg."""
    board_svg = chess.svg.board(board)
    svg_data = BytesIO(board_svg.encode('utf-8'))
    board_image = Image.open(svg_data)
    board_image = board_image.resize((400, 400))
    board_photo = ImageTk.PhotoImage(board_image)
    canvas.image = board_photo
    canvas.create_image(0, 0, anchor=tk.NW, image=board_photo)

class LearningPathwaysGUI:
    """GUI for visualizing learning pathways and gameplay."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ML Learning Pathways and Gameplay")

        # Create a frame for the chessboard
        self.chess_frame = tk.Frame(self.root)
        self.chess_frame.grid(row=0, column=0, padx=10, pady=10)
        self.chess_canvas = Canvas(self.chess_frame, width=400, height=400)
        self.chess_canvas.pack()

        # Create a frame for learning pathways
        self.pathways_frame = tk.Frame(self.root)
        self.pathways_frame.grid(row=0, column=1, padx=10, pady=10)

        # Initialize matplotlib figure for pathways
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.pathways_frame)
        self.canvas.get_tk_widget().pack()

        self.pathways_data = []
        self.board = chess.Board()

    def update_learning_pathways(self, accuracy):
        """Update the learning pathways plot dynamically."""
        self.pathways_data.append(accuracy)
        self.ax.clear()
        self.ax.plot(self.pathways_data, label="Validation Accuracy", marker="o")
        self.ax.set_title("Learning Pathways")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Accuracy")
        self.ax.legend()
        self.ax.grid()
        self.canvas.draw()

    def update_gameplay(self, board):
        """Update the chessboard during gameplay."""
        update_chess_board(self.chess_canvas, board)

    def run(self):
        self.root.mainloop()

# Example usage
if __name__ == "__main__":
    gui = LearningPathwaysGUI()

    # Simulate updates
    import time

    for i in range(10):
        time.sleep(1)  # Simulate processing delay
        gui.update_learning_pathways(accuracy=0.5 + i * 0.05)  # Example accuracy update
        gui.board.push(chess.Move.from_uci("e2e4"))  # Example move
        gui.update_gameplay(gui.board)

    gui.run()

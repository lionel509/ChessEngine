import os
import sys
from models.loader import load_models
from gui.chess_gui import ChessGUI

def main():
    model_directory = r"C:\Users\Lionel\Downloads\ChessEngine\iterations\iteration_1"
    print(f"\nSearching for models in: {model_directory}")
    
    try:
        models = load_models(model_directory)
        if not models:
            print("No models found. Please check model directory.")
            return
            
        # Launch GUI
        gui = ChessGUI(models)
        gui.run()
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
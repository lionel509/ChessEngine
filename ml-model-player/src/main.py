import os
import sys
import logging
from pathlib import Path

# Adjust Python path to include src directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.loader import load_models
from src.gui.chess_gui import ChessGUI

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    model_directory = r"C:\Users\Lionel\Downloads\ChessEngine\models"
    logging.info(f"Loading ensemble models from: {model_directory}")
    
    try:
        models = load_models(model_directory)
        if not models:
            logging.error("No models found. Please check model directory.")
            return
            
        logging.info(f"Loaded {len(models)} model ensembles")
        gui = ChessGUI(models)
        gui.run()
        
    except Exception as e:
        logging.error(f"Error during execution:", exc_info=True)
        return

if __name__ == "__main__":
    main()
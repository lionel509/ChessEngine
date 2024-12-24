import os
import pickle
from pathlib import Path

def find_pkl_files(directory):
    """Recursively find all .pkl files in directory"""
    pkl_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def load_models(directory):
    """Load all pickle models from directory"""
    models = {}
    
    # Validate directory exists
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return models
    
    # Find all .pkl files
    pkl_files = find_pkl_files(directory)
    if not pkl_files:
        print(f"No .pkl files found in: {directory}")
        return models
        
    # Load each model
    for model_path in pkl_files:
        model_name = Path(model_path).stem
        try:
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
                print(f"Successfully loaded: {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            
    return models
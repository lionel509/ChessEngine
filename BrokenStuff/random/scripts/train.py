import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
from utils.data_loader import load_training_data

def train_knn_model(data_path, model_save_path, n_neighbors=5):
    # Load training data
    X, y = load_training_data(data_path)
    X = X.reshape(X.shape[0], -1)  # Flatten board states for KNN

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Validate model
    accuracy = knn.score(X_val, y_val)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Save the model
    with open(model_save_path, 'wb') as file:
        pickle.dump(knn, file)

if __name__ == "__main__":
    train_knn_model("data/training_games.csv", "data/models/knn_model.pkl")

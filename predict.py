import os
import time
import pickle
import numpy as np
import pandas as pd

from src.banzhaf import exact_banzhaf, NUM_AGENTS
from src.features import create_features_for_one_game, clean_prediction
from src.nn import NumpyMLP
from src.plots import save_prediction_chart

def main():
    example_weights = np.array([4, 2, 7, 1, 5])
    example_quota = 10

    # Ground Truth via Exact Banzhaf [cite: 75, 77]
    real_banzhaf = exact_banzhaf(example_weights, example_quota)
    example_features = create_features_for_one_game(example_weights, example_quota)

    # Load All Models
    mlp_model, mean, std = NumpyMLP.load("models/mlp_numpy_2d.npz")
    with open("models/random_forest_scratch.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("models/extra_trees_scratch.pkl", "rb") as f:
        extra_model = pickle.load(f)

    # 1. Neural Network Prediction (Needs Scaling)
    scaled_features = (example_features - mean) / std
    mlp_prediction = clean_prediction(mlp_model.predict(np.array([scaled_features]))[0])

    # 2. Forest Predictions
    rf_prediction = clean_prediction(rf_model.predict(np.array([example_features]))[0])
    extra_prediction = clean_prediction(extra_model.predict(np.array([example_features]))[0])

    # Display results
    table = pd.DataFrame({
        "Agent": [f"Agent {i}" for i in range(NUM_AGENTS)],
        "Real": np.round(real_banzhaf, 4),
        "NN": np.round(mlp_prediction, 4),
        "RF": np.round(rf_prediction, 4),
        "Extra": np.round(extra_prediction, 4),
    })

    print("\n--- Midterm Example Prediction ---")
    print(table.to_string(index=False))

    save_prediction_chart(real_banzhaf, {"NN": mlp_prediction, "RF": rf_prediction, "Extra": extra_prediction}, "results/midterm_comparison.png")
    print("\nChart saved to results/midterm_comparison.png")

if __name__ == "__main__":
    main()
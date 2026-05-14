import os
import time
import pickle
import numpy as np
import pandas as pd

from src.banzhaf import exact_banzhaf, NUM_AGENTS
from src.features import create_features, clean_prediction
from src.nn import NumpyMLP
from src.plots import save_prediction_chart


def main():
    example_weights = np.array([4, 2, 7, 1, 5])
    example_quota = 10

    real_banzhaf = exact_banzhaf(example_weights, example_quota)
    example_features = create_features(example_weights, example_quota)

    mlp_model, mean, std = NumpyMLP.load("models/mlp_numpy.npz")
    with open("models/random_forest_scratch.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("models/extra_trees_scratch.pkl", "rb") as f:
        extra_model = pickle.load(f)

    scaled_features = (example_features - mean) / std

    start = time.perf_counter()
    mlp_prediction = mlp_model.predict(np.array([scaled_features]))[0]
    mlp_time = time.perf_counter() - start
    mlp_prediction = clean_prediction(mlp_prediction)

    start = time.perf_counter()
    rf_prediction = rf_model.predict(np.array([example_features]))[0]
    rf_time = time.perf_counter() - start
    rf_prediction = clean_prediction(rf_prediction)

    start = time.perf_counter()
    extra_prediction = extra_model.predict(np.array([example_features]))[0]
    extra_time = time.perf_counter() - start
    extra_prediction = clean_prediction(extra_prediction)

    table = pd.DataFrame({
        "Agent": [f"Agent {i}" for i in range(NUM_AGENTS)],
        "Real": np.round(real_banzhaf, 4),
        "Neural Network": np.round(mlp_prediction, 4),
        "Random Forest": np.round(rf_prediction, 4),
        "Extra Trees": np.round(extra_prediction, 4),
    })

    error_table = pd.DataFrame({
        "Model": ["Neural Network", "Random Forest", "Extra Trees"],
        "Mean Absolute Error": [
            np.mean(np.abs(real_banzhaf - mlp_prediction)),
            np.mean(np.abs(real_banzhaf - rf_prediction)),
            np.mean(np.abs(real_banzhaf - extra_prediction)),
        ],
        "Prediction Time Seconds": [mlp_time, rf_time, extra_time],
    })

    print("\nExample Prediction")
    print("------------------")
    print("Weights:       ", example_weights)
    print("Quota:         ", example_quota)
    print("Real:          ", np.round(real_banzhaf, 4))
    print("Neural Network:", np.round(mlp_prediction, 4))
    print("Random Forest: ", np.round(rf_prediction, 4))
    print("Extra Trees:   ", np.round(extra_prediction, 4))

    print("\nComparison Table")
    print("----------------")
    print(table.to_string(index=False))

    print("\nEvaluation")
    print("----------")
    print(error_table.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/example_prediction_table.csv", index=False)
    error_table.to_csv("results/example_prediction_errors.csv", index=False)

    chart_path = "results/example_prediction_comparison.png"
    save_prediction_chart(
        real_banzhaf,
        {
            "Neural Network": mlp_prediction,
            "Random Forest": rf_prediction,
            "Extra Trees": extra_prediction,
        },
        chart_path,
    )

    print("\nChart saved")
    print("-----------")
    print(chart_path)


if __name__ == "__main__":
    main()

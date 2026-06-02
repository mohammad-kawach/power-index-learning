import argparse
import os
import pickle

import numpy as np
import pandas as pd

from src.banzhaf import exact_banzhaf, NUM_AGENTS
from src.features import create_features_for_one_game, clean_prediction
from src.nn import NumpyMLP
from src.plots import save_prediction_chart

DEFAULT_MODELS_DIR = "models"
DEFAULT_RESULTS_DIR = "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare exact and predicted Banzhaf values for one game.")
    parser.add_argument(
        "--weights",
        nargs=NUM_AGENTS,
        type=float,
        default=[4, 2, 7, 1, 5],
        metavar="W",
        help=f"Exactly {NUM_AGENTS} agent weights.",
    )
    parser.add_argument("--quota", type=float, default=10, help="Voting quota.")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="Directory containing saved models.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory for saved prediction artifacts.")
    return parser.parse_args()


def require_model_files(models_dir):
    paths = {
        "mlp": os.path.join(models_dir, "mlp_numpy_2d.npz"),
        "rf": os.path.join(models_dir, "random_forest_scratch.pkl"),
        "extra": os.path.join(models_dir, "extra_trees_scratch.pkl"),
    }
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing model files. Run train_models.py first:\n{missing_text}")
    return paths


def main(args=None):
    if args is None:
        args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    example_weights = np.array(args.weights, dtype=float)
    example_quota = args.quota

    real_banzhaf = exact_banzhaf(example_weights, example_quota)
    example_features = create_features_for_one_game(example_weights, example_quota)

    paths = require_model_files(args.models_dir)
    mlp_model, mean, std = NumpyMLP.load(paths["mlp"])
    with open(paths["rf"], "rb") as f:
        rf_model = pickle.load(f)
    with open(paths["extra"], "rb") as f:
        extra_model = pickle.load(f)

    scaled_features = (example_features - mean) / std
    mlp_prediction = clean_prediction(mlp_model.predict(np.array([scaled_features]))[0])

    rf_prediction = clean_prediction(rf_model.predict(np.array([example_features]))[0])
    extra_prediction = clean_prediction(extra_model.predict(np.array([example_features]))[0])

    table = pd.DataFrame({
        "Agent": [f"Agent {i}" for i in range(NUM_AGENTS)],
        "Real": np.round(real_banzhaf, 4),
        "NN": np.round(mlp_prediction, 4),
        "RF": np.round(rf_prediction, 4),
        "Extra": np.round(extra_prediction, 4),
    })

    errors = pd.DataFrame({
        "Agent": [f"Agent {i}" for i in range(NUM_AGENTS)],
        "NN_abs_error": np.round(np.abs(mlp_prediction - real_banzhaf), 4),
        "RF_abs_error": np.round(np.abs(rf_prediction - real_banzhaf), 4),
        "Extra_abs_error": np.round(np.abs(extra_prediction - real_banzhaf), 4),
    })

    print("\n--- Midterm Example Prediction ---")
    print(table.to_string(index=False))

    table_path = os.path.join(args.results_dir, "example_prediction_table.csv")
    errors_path = os.path.join(args.results_dir, "example_prediction_errors.csv")
    chart_path = os.path.join(args.results_dir, "midterm_comparison.png")
    table.to_csv(table_path, index=False)
    errors.to_csv(errors_path, index=False)
    save_prediction_chart(
        real_banzhaf,
        {"NN": mlp_prediction, "RF": rf_prediction, "Extra": extra_prediction},
        chart_path,
    )
    print(f"\nPrediction table saved to {table_path}")
    print(f"Prediction errors saved to {errors_path}")
    print(f"Chart saved to {chart_path}")

if __name__ == "__main__":
    main()

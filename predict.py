"""Compare exact indices with all trained predictors for one voting game."""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from src.banzhaf import exact_power_indices
from src.features import clean_prediction, create_features_for_one_game
from src.nn import NumpyMLP
from src.plots import save_per_agent_error_chart, save_prediction_chart


MODEL_FILES = {
    "Scratch Random Forest": "{index}_scratch_random_forest.pkl",
    "Scratch Extra Trees": "{index}_scratch_extra_trees.pkl",
    "sklearn MLP": "{index}_sklearn_mlp.pkl",
    "sklearn Random Forest": "{index}_sklearn_random_forest.pkl",
    "sklearn Extra Trees": "{index}_sklearn_extra_trees.pkl",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=float, default=[4, 2, 7, 1, 5, 3, 6, 2])
    parser.add_argument("--quota", type=float, default=16)
    parser.add_argument("--indices", nargs="+", choices=("banzhaf", "shapley"), default=["banzhaf", "shapley"])
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def _load_predictions(index_name, features, models_dir):
    mlp_path = os.path.join(models_dir, f"{index_name}_numpy_mlp.npz")
    required = [mlp_path] + [
        os.path.join(models_dir, template.format(index=index_name))
        for template in MODEL_FILES.values()
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing model files; run train_models.py first:\n" + "\n".join(f"- {path}" for path in missing)
        )

    mlp, mean, std = NumpyMLP.load(mlp_path)
    predictions = {
        "Scratch MLP": clean_prediction(mlp.predict(np.array([(features - mean) / std]))[0])
    }
    for label, template in MODEL_FILES.items():
        path = os.path.join(models_dir, template.format(index=index_name))
        with open(path, "rb") as handle:
            artifact = pickle.load(handle)
        if label.startswith("sklearn"):
            model = artifact["model"]
            model_features = (features - mean) / std if artifact["scaled"] else features
        else:
            model = artifact
            model_features = features
        predictions[label] = clean_prediction(model.predict(np.array([model_features]))[0])
    return predictions


def main(args=None):
    args = parse_args() if args is None else args
    weights = np.asarray(args.weights, dtype=float)
    if weights.size < 2:
        raise ValueError("provide at least two weights")
    os.makedirs(args.results_dir, exist_ok=True)
    features = create_features_for_one_game(weights, args.quota)
    exact_banzhaf, exact_shapley = exact_power_indices(weights, args.quota)
    exact_indices = {"banzhaf": exact_banzhaf, "shapley": exact_shapley}

    for index_name in args.indices:
        exact = exact_indices[index_name]
        predictions = _load_predictions(index_name, features, args.models_dir)
        table_data = {"Agent": [f"Agent {i}" for i in range(weights.size)], "Exact": exact}
        table_data.update(predictions)
        table = pd.DataFrame(table_data)
        print(f"\n=== {index_name.replace('_', ' ').title()} ===")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

        prefix = os.path.join(args.results_dir, f"example_{index_name}")
        table.to_csv(f"{prefix}_predictions.csv", index=False)
        errors = pd.DataFrame(
            {"Agent": table_data["Agent"], **{label: np.abs(value - exact) for label, value in predictions.items()}}
        )
        errors.to_csv(f"{prefix}_errors.csv", index=False)
        save_prediction_chart(
            exact,
            predictions,
            f"{prefix}_comparison.png",
            title=f"Exact vs Predicted {index_name.replace('_', ' ').title()}",
            ylabel="Normalized power",
        )
        save_per_agent_error_chart(
            exact,
            predictions,
            f"{prefix}_errors.png",
            title=f"Example {index_name.replace('_', ' ').title()} Absolute Error",
            ylabel="Absolute error",
        )


if __name__ == "__main__":
    main()

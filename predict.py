"""Compare exact indices with all trained predictors for one game."""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from src.banzhaf import exact_power_indices
from src.features import clean_prediction, create_features_for_one_game, create_features_for_one_mcn
from src.mcn import (
    exact_power_indices as exact_mcn_power_indices,
    generate_random_rules,
    infer_num_agents_from_rules,
)
from src.nn import NumpyMLP
from src.plots import (
    save_mcn_rule_heatmap,
    save_per_agent_error_chart,
    save_power_index_comparison,
    save_prediction_chart,
)


MODEL_SUFFIXES = {
    "Scratch Random Forest": "scratch_random_forest.pkl",
    "Scratch Extra Trees": "scratch_extra_trees.pkl",
    "sklearn MLP": "sklearn_mlp.pkl",
    "sklearn Random Forest": "sklearn_random_forest.pkl",
    "sklearn Extra Trees": "sklearn_extra_trees.pkl",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-type", choices=("weighted", "mcn"), default="weighted")
    parser.add_argument("--weights", nargs="+", type=float, default=[4, 2, 7, 1, 5, 3, 6, 2])
    parser.add_argument("--quota", type=float, default=16)
    parser.add_argument("--indices", nargs="+", choices=("banzhaf", "shapley"), default=["banzhaf", "shapley"])
    parser.add_argument("--data", default=None, help="MCN .npz dataset to draw an example from.")
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--num-rules", type=int, default=20)
    parser.add_argument("--rule-generator", choices=("uniform", "coin_flip", "gaussian_mixture"), default="uniform")
    parser.add_argument("--value-generator", choices=("uniform", "low_variance", "high_variance"), default="uniform")
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--num-coins", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def _artifact_prefix(game_type, index_name):
    return index_name if game_type == "weighted" else f"{game_type}_{index_name}"


def _load_predictions(index_name, features, models_dir, game_type):
    prefix = _artifact_prefix(game_type, index_name)
    mlp_path = os.path.join(models_dir, f"{prefix}_numpy_mlp.npz")
    required = [mlp_path] + [os.path.join(models_dir, f"{prefix}_{suffix}") for suffix in MODEL_SUFFIXES.values()]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing model files; run train_models.py first:\n" + "\n".join(f"- {path}" for path in missing)
        )

    mlp, mean, std = NumpyMLP.load(mlp_path)
    if features.shape[0] != mean.shape[0]:
        raise ValueError(
            f"example has {features.shape[0]} features, but the model expects {mean.shape[0]}. "
            "Use the same game type, agent count, and MCN rule count used for training."
        )
    predictions = {
        "Scratch MLP": clean_prediction(mlp.predict(np.array([(features - mean) / std]))[0])
    }
    for label, suffix in MODEL_SUFFIXES.items():
        path = os.path.join(models_dir, f"{prefix}_{suffix}")
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


def _mcn_rule_table(rules):
    num_agents = infer_num_agents_from_rules(rules)
    columns = [f"req_{i}" for i in range(num_agents)]
    columns += [f"ban_{i}" for i in range(num_agents)]
    columns += ["value"]
    return pd.DataFrame(rules, columns=columns)


def _prepare_mcn_example(args):
    if args.data:
        archive = np.load(args.data, allow_pickle=False)
        rules = archive["rules"][args.example_index]
    else:
        rng = np.random.default_rng(args.seed)
        rules = generate_random_rules(
            args.num_rules,
            args.num_agents,
            rng=rng,
            rule_generator=args.rule_generator,
            value_generator=args.value_generator,
            p=args.p,
            num_coins=args.num_coins,
        )
    result = exact_mcn_power_indices(rules)
    features = create_features_for_one_mcn(rules)
    exact_indices = {"banzhaf": result.banzhaf, "shapley": result.shapley}
    return features, exact_indices, _mcn_rule_table(rules)


def main(args=None):
    args = parse_args() if args is None else args
    os.makedirs(args.results_dir, exist_ok=True)
    if args.game_type == "mcn":
        features, exact_indices, rule_table = _prepare_mcn_example(args)
        rule_table_path = os.path.join(args.results_dir, "example_mcn_rules.csv")
        rule_chart_path = os.path.join(args.results_dir, "example_mcn_rules.png")
        exact_power_chart_path = os.path.join(args.results_dir, "example_mcn_exact_power.png")
        rule_table.to_csv(rule_table_path, index=False)
        save_mcn_rule_heatmap(rule_table.values, rule_chart_path)
        save_power_index_comparison(
            exact_indices["banzhaf"],
            exact_indices["shapley"],
            exact_power_chart_path,
            title="Exact Power Indices for the Example MCN",
        )
        agent_count = len(exact_indices["banzhaf"])
        output_stem = "example_mcn"
        print(
            "Saved MCN rule table and charts to "
            f"{rule_table_path}, {rule_chart_path}, and {exact_power_chart_path}"
        )
    else:
        weights = np.asarray(args.weights, dtype=float)
        if weights.size < 2:
            raise ValueError("provide at least two weights")
        features = create_features_for_one_game(weights, args.quota)
        exact_banzhaf, exact_shapley = exact_power_indices(weights, args.quota)
        exact_indices = {"banzhaf": exact_banzhaf, "shapley": exact_shapley}
        agent_count = weights.size
        output_stem = "example"

    for index_name in args.indices:
        exact = exact_indices[index_name]
        predictions = _load_predictions(index_name, features, args.models_dir, args.game_type)
        table_data = {"Agent": [f"Agent {i}" for i in range(agent_count)], "Exact": exact}
        table_data.update(predictions)
        table = pd.DataFrame(table_data)
        print(f"\n=== {index_name.replace('_', ' ').title()} ===")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

        prefix = os.path.join(args.results_dir, f"{output_stem}_{index_name}")
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

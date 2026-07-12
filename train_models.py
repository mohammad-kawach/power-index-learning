"""Train from-scratch and scikit-learn power-index predictors."""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from src.features import (
    clean_prediction,
    create_feature_matrix,
    create_mcn_feature_matrix,
    infer_num_agents,
    target_columns,
)
from src.nn import NumpyMLP
from src.plots import (
    save_loss_curve,
    save_model_mae_chart,
    save_per_agent_error_chart,
    save_prediction_scatter,
)
from src.scaler import StandardScalerScratch
from src.trees import ForestRegressorScratch


RANDOM_SEED = 42


def mae(prediction, target):
    return float(np.mean(np.abs(prediction - target)))


def clean_batch(predictions):
    return np.array([clean_prediction(row) for row in predictions])


def _save_pickle(value, path):
    with open(path, "wb") as handle:
        pickle.dump(value, handle)


def _load_sklearn_models(args):
    try:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required for baseline comparisons; run "
            "'python -m pip install -r requirements.txt'"
        ) from error

    hidden_layer_sizes = (512, 256, 128) if args.game_type == "mcn" else (128, 64)
    return {
        "sklearn MLP": MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=args.sklearn_max_iter,
            batch_size=args.batch_size,
            random_state=RANDOM_SEED,
            early_stopping=True,
        ),
        "sklearn Random Forest": RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "sklearn Extra Trees": ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_depth=11,
            random_state=RANDOM_SEED + 1,
            n_jobs=-1,
        ),
    }


def _artifact_name(args, index_name, suffix):
    prefix = index_name if args.game_type == "weighted" else f"{args.game_type}_{index_name}"
    return os.path.join(args.models_dir, f"{prefix}_{suffix}")


def _result_name(args, filename):
    if args.game_type == "weighted":
        return os.path.join(args.results_dir, filename)
    return os.path.join(args.results_dir, f"{args.game_type}_{filename}")


def _load_dataset(args):
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"{args.data} not found; run generate_data.py first")

    if args.game_type == "mcn":
        archive = np.load(args.data, allow_pickle=False)
        rules = archive["rules"]
        X = create_mcn_feature_matrix(rules)
        num_agents = (rules.shape[2] - 1) // 2
        targets = {
            "banzhaf": archive["banzhaf_targets"].astype(float),
            "shapley": archive["shapley_targets"].astype(float),
        }
        return X, targets, num_agents

    dataframe = pd.read_csv(args.data)
    if len(dataframe) < 20:
        raise ValueError("training requires at least 20 generated games")
    num_agents = infer_num_agents(dataframe)
    X = create_feature_matrix(dataframe)
    targets = {}
    for index_name in args.indices:
        columns = target_columns(dataframe, index_name)
        if len(columns) != num_agents:
            raise ValueError(
                f"dataset needs {num_agents} {index_name}_target_* columns; regenerate it with generate_data.py"
            )
        targets[index_name] = dataframe[columns].values.astype(float)
    return X, targets, num_agents


def _train_one_index(index_name, X_train, X_test, y_train, y_test, scaler, args):
    title = "Banzhaf" if index_name == "banzhaf" else "Shapley--Shubik"
    if args.game_type == "mcn":
        title = f"MCN {title}"
    print(f"\n=== {title} predictors ===")
    predictions = {}
    metric_rows = []

    if args.game_type == "mcn":
        mlp = NumpyMLP(
            input_size=X_train.shape[1],
            hidden_layers=(512, 256, 128),
            output_size=y_train.shape[1],
            learning_rate=0.001,
            seed=RANDOM_SEED,
            output_activation="linear",
            loss="mse",
            dropout_rate=0.2,
        )
    else:
        mlp = NumpyMLP(
            input_size=X_train.shape[1],
            hidden1=128,
            hidden2=64,
            output_size=y_train.shape[1],
            learning_rate=0.001,
            seed=RANDOM_SEED,
        )
    history = mlp.train(
        scaler.transform(X_train),
        y_train,
        scaler.transform(X_test),
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        print_every=max(1, args.epochs // 10),
    )
    mlp.save(_artifact_name(args, index_name, "numpy_mlp.npz"), scaler)
    history_path = _result_name(args, f"{index_name}_numpy_mlp_history.csv")
    pd.DataFrame(history, columns=["epoch", "train_loss", "test_mae"]).to_csv(
        history_path, index=False
    )
    save_loss_curve(
        history,
        _result_name(args, f"{index_name}_numpy_mlp_training.png"),
        title=f"{title}: NumPy MLP Training",
    )
    predictions["Scratch MLP"] = clean_batch(mlp.predict(scaler.transform(X_test)))

    scratch_models = {
        "Scratch Random Forest": ForestRegressorScratch(
            n_estimators=args.n_estimators,
            max_depth=10,
            mode="random_forest",
            seed=RANDOM_SEED,
            verbose=args.verbose,
        ),
        "Scratch Extra Trees": ForestRegressorScratch(
            n_estimators=args.n_estimators,
            max_depth=11,
            mode="extra_trees",
            seed=RANDOM_SEED + 1,
            verbose=args.verbose,
        ),
    }
    for label, model in scratch_models.items():
        print(f"Training {label}...")
        model.fit(X_train, y_train)
        slug = "random_forest" if "Random" in label else "extra_trees"
        _save_pickle(model, _artifact_name(args, index_name, f"scratch_{slug}.pkl"))
        predictions[label] = clean_batch(model.predict(X_test))

    for label, model in _load_sklearn_models(args).items():
        print(f"Training {label}...")
        is_mlp = label == "sklearn MLP"
        train_features = scaler.transform(X_train) if is_mlp else X_train
        test_features = scaler.transform(X_test) if is_mlp else X_test
        model.fit(train_features, y_train)
        slug = label.removeprefix("sklearn ").lower().replace(" ", "_")
        artifact = {"model": model, "scaled": is_mlp, "game_type": args.game_type}
        _save_pickle(artifact, _artifact_name(args, index_name, f"sklearn_{slug}.pkl"))
        predictions[label] = clean_batch(model.predict(test_features))

    for label, values in predictions.items():
        metric_rows.append(
            {
                "index": title.replace("--", "-"),
                "game_type": args.game_type,
                "implementation": "scikit-learn" if label.startswith("sklearn") else "from scratch",
                "model": label,
                "test_mae": mae(values, y_test),
            }
        )

    save_prediction_scatter(
        y_test,
        predictions,
        _result_name(args, f"{index_name}_prediction_scatter.png"),
        title=f"{title}: Predicted vs Exact",
    )
    save_per_agent_error_chart(
        y_test,
        predictions,
        _result_name(args, f"{index_name}_per_agent_mae.png"),
        title=f"{title}: Per-Agent Test MAE",
    )
    return metric_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/voting_games.csv")
    parser.add_argument("--game-type", choices=("auto", "weighted", "mcn"), default="auto")
    parser.add_argument("--indices", nargs="+", choices=("banzhaf", "shapley"), default=["banzhaf", "shapley"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--sklearn-max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-estimators", type=int, default=40)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--verbose", action="store_true", help="Print progress for every scratch tree.")
    return parser.parse_args()


def main(args=None):
    args = parse_args() if args is None else args
    for name in ("epochs", "sklearn_max_iter", "batch_size", "n_estimators"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.game_type == "auto":
        args.game_type = "mcn" if args.data.endswith(".npz") else "weighted"
    # Fail before expensive from-scratch training if the comparison dependency
    # was not installed.
    _load_sklearn_models(args)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    X, targets, num_agents = _load_dataset(args)
    if len(X) < 20:
        raise ValueError("training requires at least 20 generated games")
    for index_name in args.indices:
        if index_name not in targets or targets[index_name].shape[1] != num_agents:
            raise ValueError(f"dataset needs {num_agents} {index_name} target columns")

    rng = np.random.default_rng(RANDOM_SEED)
    permutation = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_indices, test_indices = permutation[:split], permutation[split:]
    X_train, X_test = X[train_indices], X[test_indices]
    scaler = StandardScalerScratch().fit(X_train)

    rows = []
    for index_name in args.indices:
        y = targets[index_name]
        rows.extend(
            _train_one_index(
                index_name,
                X_train,
                X_test,
                y[train_indices],
                y[test_indices],
                scaler,
                args,
            )
        )

    metrics = pd.DataFrame(rows).sort_values(["index", "test_mae"])
    metrics_path = _result_name(args, "model_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    save_model_mae_chart(metrics, _result_name(args, "model_mae_comparison.png"))
    print("\n=== Final test MAE ===")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved metrics to {metrics_path}")
    return metrics


if __name__ == "__main__":
    main()

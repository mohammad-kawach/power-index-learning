"""Train from-scratch and scikit-learn power-index predictors."""

import argparse
import os
import pickle
from itertools import product

import numpy as np
import pandas as pd

from src.features import (
    MCN_FEATURE_SETS,
    clean_prediction,
    create_feature_matrix,
    create_mcn_feature_matrix,
    infer_num_agents,
    target_columns,
)
from src.nn import NumpyMLP
from src.plots import (
    save_loss_curve,
    save_mcn_agent_role_chart,
    save_mcn_index_relationship_chart,
    save_mcn_rule_complexity_chart,
    save_mcn_target_distribution_chart,
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


def _load_sklearn_models(args, overrides=None):
    try:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required for baseline comparisons; run "
            "'python -m pip install -r requirements.txt'"
        ) from error

    overrides = overrides or {}
    hidden_layer_sizes = (512, 256, 128) if args.game_type == "mcn" else (128, 64)
    model_specs = {
        "sklearn MLP": (
            MLPRegressor,
            {
                "hidden_layer_sizes": hidden_layer_sizes,
                "max_iter": args.sklearn_max_iter,
                "batch_size": args.batch_size,
                "random_state": RANDOM_SEED,
                "early_stopping": True,
            },
        ),
        "sklearn Random Forest": (
            RandomForestRegressor,
            {
                "n_estimators": args.n_estimators,
                "max_depth": 10,
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
            },
        ),
        "sklearn Extra Trees": (
            ExtraTreesRegressor,
            {
                "n_estimators": args.n_estimators,
                "max_depth": 11,
                "random_state": RANDOM_SEED + 1,
                "n_jobs": -1,
            },
        ),
    }
    models = {}
    for label, (model_class, params) in model_specs.items():
        params = {**params, **overrides.get(label, {})}
        models[label] = model_class(**params)
    return models


def _scratch_model_specs(args, overrides=None, *, verbose=None):
    overrides = overrides or {}
    if verbose is None:
        verbose = args.verbose
    specs = {
        "Scratch Random Forest": {
            "n_estimators": args.n_estimators,
            "max_depth": 10,
            "min_samples_split": 8,
            "max_features": "sqrt",
            "mode": "random_forest",
            "seed": RANDOM_SEED,
            "verbose": verbose,
        },
        "Scratch Extra Trees": {
            "n_estimators": args.n_estimators,
            "max_depth": 11,
            "min_samples_split": 8,
            "max_features": "sqrt",
            "mode": "extra_trees",
            "seed": RANDOM_SEED + 1,
            "verbose": verbose,
        },
    }
    return {label: {**params, **overrides.get(label, {})} for label, params in specs.items()}


def _build_scratch_models(args, overrides=None):
    return {
        label: ForestRegressorScratch(**params)
        for label, params in _scratch_model_specs(args, overrides).items()
    }


def _iter_parameter_grid(grid):
    keys = list(grid)
    for values in product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def _format_params(params):
    if not params:
        return ""
    return ", ".join(f"{key}={value!r}" for key, value in sorted(params.items()))


def _tune_sklearn_ensembles(args, X_train, y_train, X_validation, y_validation):
    if X_validation is None:
        return {}

    tuning_grids = {
        "sklearn Random Forest": {
            "max_depth": [8, 10, 12, 14],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [1.0, "sqrt"],
        },
        "sklearn Extra Trees": {
            "max_depth": [9, 11, 13, 15],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [1.0, "sqrt"],
        },
    }
    best_overrides = {}
    for label, grid in tuning_grids.items():
        print(f"Tuning {label} on validation data...")
        best_score = float("inf")
        best_params = None
        for params in _iter_parameter_grid(grid):
            candidate_params = {**params, "n_estimators": args.tuning_estimators}
            model = _load_sklearn_models(args, {label: candidate_params})[label]
            model.fit(X_train, y_train)
            score = mae(clean_batch(model.predict(X_validation)), y_validation)
            if score < best_score:
                best_score = score
                best_params = params
        best_overrides[label] = best_params
        print(f"  selected {_format_params(best_params)} | validation MAE={best_score:.6f}")
    return best_overrides


def _tune_scratch_ensembles(args, X_train, y_train, X_validation, y_validation):
    if X_validation is None:
        return {}

    tuning_grids = {
        "Scratch Random Forest": {
            "max_depth": [8, 10, 12],
            "min_samples_split": [4, 8],
            "max_features": ["sqrt", None],
        },
        "Scratch Extra Trees": {
            "max_depth": [9, 11, 13],
            "min_samples_split": [4, 8],
            "max_features": ["sqrt", None],
        },
    }
    best_overrides = {}
    for label, grid in tuning_grids.items():
        print(f"Tuning {label} on validation data...")
        best_score = float("inf")
        best_params = None
        for params in _iter_parameter_grid(grid):
            candidate_params = {**params, "n_estimators": args.tuning_estimators}
            specs = _scratch_model_specs(args, {label: candidate_params}, verbose=False)
            model = ForestRegressorScratch(**specs[label])
            model.fit(X_train, y_train)
            score = mae(clean_batch(model.predict(X_validation)), y_validation)
            if score < best_score:
                best_score = score
                best_params = params
        best_overrides[label] = best_params
        print(f"  selected {_format_params(best_params)} | validation MAE={best_score:.6f}")
    return best_overrides


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
        X = create_mcn_feature_matrix(rules, feature_set=args.mcn_feature_set)
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


def _save_mcn_dataset_plots(args, targets):
    if args.game_type != "mcn":
        return
    archive = np.load(args.data, allow_pickle=False)
    rules = archive["rules"]
    save_mcn_rule_complexity_chart(
        rules,
        _result_name(args, "rule_complexity.png"),
    )
    save_mcn_agent_role_chart(
        rules,
        _result_name(args, "agent_role_frequency.png"),
    )
    save_mcn_target_distribution_chart(
        targets["banzhaf"],
        targets["shapley"],
        _result_name(args, "target_distribution.png"),
    )
    save_mcn_index_relationship_chart(
        targets["banzhaf"],
        targets["shapley"],
        _result_name(args, "index_relationship.png"),
    )


def _train_one_index(
    index_name,
    X_train,
    X_validation,
    X_test,
    y_train,
    y_validation,
    y_test,
    scaler,
    args,
):
    title = "Banzhaf" if index_name == "banzhaf" else "Shapley--Shubik"
    if args.game_type == "mcn":
        title = f"MCN {title}"
    print(f"\n=== {title} predictors ===")
    predictions = {}
    validation_predictions = {}
    model_params = {}
    metric_rows = []
    evaluation_features = X_validation if X_validation is not None else X_test
    evaluation_target = y_validation if y_validation is not None else y_test
    evaluation_label = "Validation MAE" if X_validation is not None else "Test MAE"

    sklearn_overrides = {}
    scratch_overrides = {}
    if args.tune_ensembles:
        sklearn_overrides = _tune_sklearn_ensembles(
            args, X_train, y_train, X_validation, y_validation
        )
    if args.tune_scratch_ensembles and not args.skip_scratch_ensembles:
        scratch_overrides = _tune_scratch_ensembles(
            args, X_train, y_train, X_validation, y_validation
        )

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
        scaler.transform(evaluation_features),
        evaluation_target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        print_every=max(1, args.epochs // 10),
        evaluation_label=evaluation_label,
    )
    mlp_metadata = {"game_type": args.game_type}
    if args.game_type == "mcn":
        mlp_metadata["mcn_feature_set"] = args.mcn_feature_set
    mlp.save(_artifact_name(args, index_name, "numpy_mlp.npz"), scaler, mlp_metadata)
    history_path = _result_name(args, f"{index_name}_numpy_mlp_history.csv")
    pd.DataFrame(history, columns=["epoch", "train_loss", "evaluation_mae"]).to_csv(
        history_path, index=False
    )
    save_loss_curve(
        history,
        _result_name(args, f"{index_name}_numpy_mlp_training.png"),
        title=f"{title}: NumPy MLP Training",
        metric_label=evaluation_label,
    )
    predictions["Scratch MLP"] = clean_batch(mlp.predict(scaler.transform(X_test)))
    if X_validation is not None:
        validation_predictions["Scratch MLP"] = clean_batch(
            mlp.predict(scaler.transform(X_validation))
        )

    if not args.skip_scratch_ensembles:
        scratch_models = _build_scratch_models(args, scratch_overrides)
        for label, model in scratch_models.items():
            print(f"Training {label}...")
            model.fit(X_train, y_train)
            slug = "random_forest" if "Random" in label else "extra_trees"
            _save_pickle(model, _artifact_name(args, index_name, f"scratch_{slug}.pkl"))
            model_params[label] = scratch_overrides.get(label, {})
            predictions[label] = clean_batch(model.predict(X_test))
            if X_validation is not None:
                validation_predictions[label] = clean_batch(model.predict(X_validation))

    for label, model in _load_sklearn_models(args, sklearn_overrides).items():
        print(f"Training {label}...")
        is_mlp = label == "sklearn MLP"
        train_features = scaler.transform(X_train) if is_mlp else X_train
        test_features = scaler.transform(X_test) if is_mlp else X_test
        validation_features = (
            scaler.transform(X_validation)
            if is_mlp and X_validation is not None
            else X_validation
        )
        model.fit(train_features, y_train)
        slug = label.removeprefix("sklearn ").lower().replace(" ", "_")
        artifact = {"model": model, "scaled": is_mlp, "game_type": args.game_type}
        if args.game_type == "mcn":
            artifact["mcn_feature_set"] = args.mcn_feature_set
        _save_pickle(artifact, _artifact_name(args, index_name, f"sklearn_{slug}.pkl"))
        model_params[label] = sklearn_overrides.get(label, {})
        predictions[label] = clean_batch(model.predict(test_features))
        if X_validation is not None:
            validation_predictions[label] = clean_batch(model.predict(validation_features))

    for label, values in predictions.items():
        validation_mae = (
            mae(validation_predictions[label], y_validation)
            if X_validation is not None
            else np.nan
        )
        metric_rows.append(
            {
                "index": title.replace("--", "-"),
                "game_type": args.game_type,
                "feature_set": args.mcn_feature_set if args.game_type == "mcn" else "",
                "implementation": "scikit-learn" if label.startswith("sklearn") else "from scratch",
                "model": label,
                "validation_mae": validation_mae,
                "test_mae": mae(values, y_test),
                "selected_params": _format_params(model_params.get(label, {})),
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
    parser.add_argument(
        "--mcn-feature-set",
        choices=MCN_FEATURE_SETS,
        default="augmented",
        help="MCN input features: raw flattened rules or raw rules plus aggregate summaries.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--sklearn-max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-estimators", type=int, default=40)
    parser.add_argument(
        "--tuning-estimators",
        type=int,
        default=20,
        help="Trees per candidate during ensemble tuning; final models still use --n-estimators.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.0,
        help="Fraction of the non-test split reserved for validation. Tuning uses 0.2 if this is 0.",
    )
    parser.add_argument(
        "--tune-ensembles",
        action="store_true",
        help="Grid-search scikit-learn tree ensembles on the validation split.",
    )
    parser.add_argument(
        "--tune-scratch-ensembles",
        action="store_true",
        help="Also grid-search the slower from-scratch tree ensembles.",
    )
    parser.add_argument(
        "--skip-scratch-ensembles",
        action="store_true",
        help="Skip the slower from-scratch Random Forest and Extra Trees models.",
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--verbose", action="store_true", help="Print progress for every scratch tree.")
    return parser.parse_args()


def main(args=None):
    args = parse_args() if args is None else args
    for name in (
        "epochs",
        "sklearn_max_iter",
        "batch_size",
        "n_estimators",
        "tuning_estimators",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if not 0 <= args.validation_size < 1:
        raise ValueError("--validation-size must be in [0, 1)")
    if args.skip_scratch_ensembles and args.tune_scratch_ensembles:
        raise ValueError("--skip-scratch-ensembles cannot be combined with --tune-scratch-ensembles")
    if args.tune_scratch_ensembles:
        args.tune_ensembles = True
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
    _save_mcn_dataset_plots(args, targets)

    rng = np.random.default_rng(RANDOM_SEED)
    permutation = rng.permutation(len(X))
    test_count = max(1, int(round(args.test_size * len(X))))
    if len(X) - test_count < 2:
        raise ValueError("test split leaves fewer than two training examples")
    train_pool_indices, test_indices = permutation[:-test_count], permutation[-test_count:]

    validation_size = args.validation_size
    if args.tune_ensembles and validation_size == 0:
        validation_size = 0.2
    if validation_size > 0:
        validation_count = max(1, int(round(validation_size * len(train_pool_indices))))
        if len(train_pool_indices) - validation_count < 2:
            raise ValueError("validation split leaves fewer than two training examples")
        train_indices = train_pool_indices[:-validation_count]
        validation_indices = train_pool_indices[-validation_count:]
    else:
        train_indices = train_pool_indices
        validation_indices = None

    X_train = X[train_indices]
    X_validation = X[validation_indices] if validation_indices is not None else None
    X_test = X[test_indices]
    scaler = StandardScalerScratch().fit(X_train)
    split_message = f"Split: train={len(train_indices)}, test={len(test_indices)}"
    if validation_indices is not None:
        split_message = (
            f"Split: train={len(train_indices)}, "
            f"validation={len(validation_indices)}, test={len(test_indices)}"
        )
    if args.game_type == "mcn":
        split_message += f" | feature_set={args.mcn_feature_set} | input_width={X.shape[1]}"
    print(split_message)

    rows = []
    for index_name in args.indices:
        y = targets[index_name]
        rows.extend(
            _train_one_index(
                index_name,
                X_train,
                X_validation,
                X_test,
                y[train_indices],
                y[validation_indices] if validation_indices is not None else None,
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

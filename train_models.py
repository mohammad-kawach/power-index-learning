import argparse
import os
import pickle

import numpy as np
import pandas as pd

from src.banzhaf import NUM_AGENTS
from src.features import create_feature_matrix, clean_prediction 
from src.scaler import StandardScalerScratch
from src.nn import NumpyMLP
from src.trees import ForestRegressorScratch
from src.plots import save_loss_curve

RANDOM_SEED = 42
DEFAULT_DATASET_PATH = "data/midterm_2d_data.csv"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_MODELS_DIR = "models"


def mae(pred, y):
    return float(np.mean(np.abs(pred - y)))


def clean_batch(predictions):
    return np.array([clean_prediction(p) for p in predictions])


def parse_args():
    parser = argparse.ArgumentParser(description="Train from-scratch Banzhaf prediction models.")
    parser.add_argument("--data", default=DEFAULT_DATASET_PATH, help="Input dataset CSV path.")
    parser.add_argument("--epochs", type=int, default=1000, help="Neural network training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Neural network mini-batch size.")
    parser.add_argument("--n-estimators", type=int, default=40, help="Number of trees for each forest model.")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="Directory for saved models.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory for saved metrics and plots.")
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found. Run generate_data.py first.")
        return
        
    df = pd.read_csv(args.data)
    X = create_feature_matrix(df) 
    y = df[[f"target_{i}" for i in range(NUM_AGENTS)]].values.astype(float)

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScalerScratch()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 1. Neural Network ---
    print(f"\nTraining NumPy Neural Network ({X_train_scaled.shape[1]} features)")
    mlp_model = NumpyMLP(
        input_size=X_train_scaled.shape[1],
        hidden1=128, hidden2=64, output_size=NUM_AGENTS,
        learning_rate=0.001, seed=RANDOM_SEED
    )
    history = mlp_model.train(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        print_every=100,
    )
    
    mlp_model.save(os.path.join(args.models_dir, "mlp_numpy_2d.npz"), scaler)
    save_loss_curve(history, os.path.join(args.results_dir, "mlp_training_curve_2d.png"))
    pd.DataFrame(history, columns=["epoch", "train_loss", "test_mae"]).to_csv(
        os.path.join(args.results_dir, "mlp_training_history.csv"),
        index=False,
    )

    # --- 2. Random Forest ---
    print("\nTraining From-Scratch Random Forest...")
    rf_model = ForestRegressorScratch(
        n_estimators=args.n_estimators,
        max_depth=10,
        mode="random_forest",
        seed=RANDOM_SEED,
    )
    rf_model.fit(X_train, y_train)
    
    with open(os.path.join(args.models_dir, "random_forest_scratch.pkl"), "wb") as f:
        pickle.dump(rf_model, f)

    # --- 3. Extra Trees ---
    print("Training From-Scratch Extra Trees...")
    extra_model = ForestRegressorScratch(
        n_estimators=args.n_estimators,
        max_depth=11,
        mode="extra_trees",
        seed=RANDOM_SEED + 1,
    )
    extra_model.fit(X_train, y_train)
    
    with open(os.path.join(args.models_dir, "extra_trees_scratch.pkl"), "wb") as f:
        pickle.dump(extra_model, f)

    # Evaluation
    mlp_pred = clean_batch(mlp_model.predict(X_test_scaled))
    rf_pred = clean_batch(rf_model.predict(X_test))
    extra_pred = clean_batch(extra_model.predict(X_test))

    metrics = pd.DataFrame(
        [
            {"model": "NumPy Neural Network", "test_mae": mae(mlp_pred, y_test)},
            {"model": "From-Scratch Random Forest", "test_mae": mae(rf_pred, y_test)},
            {"model": "From-Scratch Extra Trees", "test_mae": mae(extra_pred, y_test)},
        ]
    )
    metrics_path = os.path.join(args.results_dir, "model_metrics_2d.csv")
    metrics.to_csv(metrics_path, index=False)

    print("\n--- Final Test Results ---")
    print(f"NN MAE:    {metrics.loc[0, 'test_mae']:.6f}")
    print(f"RF MAE:    {metrics.loc[1, 'test_mae']:.6f}")
    print(f"Extra MAE: {metrics.loc[2, 'test_mae']:.6f}")
    print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    main()

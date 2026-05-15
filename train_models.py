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

def mae(pred, y):
    return float(np.mean(np.abs(pred - y)))

def clean_batch(predictions):
    return np.array([clean_prediction(p) for p in predictions])

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if not os.path.exists("data/midterm_2d_data.csv"):
        print("Error: data/midterm_2d_data.csv not found. Run generate_data.py first.")
        return
        
    df = pd.read_csv("data/midterm_2d_data.csv")
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
    history = mlp_model.train(X_train_scaled, y_train, X_test_scaled, y_test, epochs=1000, batch_size=256, print_every=100)
    
    # Save NN
    mlp_model.save("models/mlp_numpy_2d.npz", scaler)
    save_loss_curve(history, "results/mlp_training_curve_2d.png")

    # --- 2. Random Forest ---
    print("\nTraining From-Scratch Random Forest...")
    rf_model = ForestRegressorScratch(n_estimators=40, max_depth=10, mode="random_forest", seed=RANDOM_SEED)
    rf_model.fit(X_train, y_train)
    
    # SAVE Random Forest
    with open("models/random_forest_scratch.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    # --- 3. Extra Trees ---
    print("Training From-Scratch Extra Trees...")
    extra_model = ForestRegressorScratch(n_estimators=40, max_depth=11, mode="extra_trees", seed=RANDOM_SEED + 1)
    extra_model.fit(X_train, y_train)
    
    # SAVE Extra Trees
    with open("models/extra_trees_scratch.pkl", "wb") as f:
        pickle.dump(extra_model, f)

    # Evaluation
    mlp_pred = clean_batch(mlp_model.predict(X_test_scaled))
    rf_pred = clean_batch(rf_model.predict(X_test))
    extra_pred = clean_batch(extra_model.predict(X_test))

    print("\n--- Final Test Results ---")
    print(f"NN MAE:    {mae(mlp_pred, y_test):.6f}")
    print(f"RF MAE:    {mae(rf_pred, y_test):.6f}")
    print(f"Extra MAE: {mae(extra_pred, y_test):.6f}")

if __name__ == "__main__":
    main()
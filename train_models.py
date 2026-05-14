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

    df = pd.read_csv("data/dataset.csv")

    X = create_feature_matrix(df)
    y = df[[f"banzhaf_{i}" for i in range(NUM_AGENTS)]].values.astype(float)

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScalerScratch()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining NumPy Neural Network")
    print("-----------------------------")
    mlp_model = NumpyMLP(
        input_size=X_train_scaled.shape[1],
        hidden1=128,
        hidden2=64,
        output_size=NUM_AGENTS,
        learning_rate=0.001,
        seed=RANDOM_SEED,
    )
    history = mlp_model.train(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        epochs=800,
        batch_size=256,
        print_every=50,
    )
    mlp_model.save("models/mlp_numpy.npz", scaler)
    np.savetxt("results/mlp_training_history.csv", history, delimiter=",", header="epoch,loss,test_mae", comments="")
    save_loss_curve(history, "results/mlp_training_curve.png")

    print("\nTraining From-Scratch Random Forest")
    print("-----------------------------------")
    rf_model = ForestRegressorScratch(
        n_estimators=35,
        max_depth=9,
        min_samples_split=8,
        max_features="sqrt",
        mode="random_forest",
        seed=RANDOM_SEED,
    )
    rf_model.fit(X_train, y_train)
    with open("models/random_forest_scratch.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    print("\nTraining From-Scratch Extra Trees")
    print("---------------------------------")
    extra_model = ForestRegressorScratch(
        n_estimators=35,
        max_depth=10,
        min_samples_split=8,
        max_features="sqrt",
        mode="extra_trees",
        seed=RANDOM_SEED + 1,
    )
    extra_model.fit(X_train, y_train)
    with open("models/extra_trees_scratch.pkl", "wb") as f:
        pickle.dump(extra_model, f)

    mlp_pred = clean_batch(mlp_model.predict(X_test_scaled))
    rf_pred = clean_batch(rf_model.predict(X_test))
    extra_pred = clean_batch(extra_model.predict(X_test))

    results = pd.DataFrame({
        "Model": ["NumPy Neural Network", "From-Scratch Random Forest", "From-Scratch Extra Trees"],
        "Test MAE": [mae(mlp_pred, y_test), mae(rf_pred, y_test), mae(extra_pred, y_test)],
    })
    results.to_csv("results/model_metrics.csv", index=False)

    print("\nFinal Test Results")
    print("------------------")
    print(results.to_string(index=False))
    print("\nSaved models to models/")
    print("Saved plots and metrics to results/")


if __name__ == "__main__":
    main()

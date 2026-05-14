import os
import numpy as np
import pandas as pd

from src.banzhaf import exact_banzhaf, NUM_AGENTS

NUM_SAMPLES = 20000
RANDOM_SEED = 42


def generate_dataset(num_samples: int = NUM_SAMPLES):
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for _ in range(num_samples):
        weights = rng.integers(1, 10, size=NUM_AGENTS)

        # quota between 40% and 80% of total weight
        min_quota = max(1, int(weights.sum() * 0.4))
        max_quota = max(min_quota + 1, int(weights.sum() * 0.8))
        quota = int(rng.integers(min_quota, max_quota))

        banzhaf = exact_banzhaf(weights, quota)

        row = {f"weight_{i}": int(weights[i]) for i in range(NUM_AGENTS)}
        row["quota"] = quota
        row.update({f"banzhaf_{i}": float(banzhaf[i]) for i in range(NUM_AGENTS)})
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dataset.csv", index=False)

    print("Dataset saved to data/dataset.csv")
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    generate_dataset()

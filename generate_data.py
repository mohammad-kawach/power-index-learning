import os
import argparse

import numpy as np
import pandas as pd

from src.banzhaf import exact_banzhaf, NUM_AGENTS


def generate_midterm_2d_dataset(
    num_games=20000,
    seed=42,
    output_path="data/midterm_2d_data.csv",
):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    data = []
    print(f"Generating {num_games} 2D voting games with seed={seed}...")
    
    for _ in range(num_games):
        weights = rng.integers(1, 20, size=NUM_AGENTS)
        total = np.sum(weights)
        quota = rng.integers(int(total * 0.5), int(total * 0.8) + 1)
        
        labels = exact_banzhaf(weights, quota)
        
        row = {f"weight_{i}": weights[i] for i in range(NUM_AGENTS)}
        row["quota"] = quota
        row.update({f"target_{i}": labels[i] for i in range(NUM_AGENTS)})
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"2D dataset saved to {output_path} | Shape: {df.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate weighted voting games with exact Banzhaf labels.")
    parser.add_argument("--num-games", type=int, default=20000, help="Number of random games to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible datasets.")
    parser.add_argument("--output", default="data/midterm_2d_data.csv", help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_midterm_2d_dataset(num_games=args.num_games, seed=args.seed, output_path=args.output)

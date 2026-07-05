"""Generate weighted voting games with exact power-index labels."""

import argparse
import os

import numpy as np
import pandas as pd

from src.banzhaf import exact_power_indices


def generate_dataset(
    num_games=20_000,
    num_agents=8,
    seed=42,
    output_path="data/voting_games.csv",
):
    """Generate exact Banzhaf and Shapley--Shubik training targets."""
    if num_games <= 0:
        raise ValueError("num_games must be positive")
    if num_agents < 2:
        raise ValueError("num_agents must be at least 2")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows = []
    print(f"Generating {num_games} games with {num_agents} agents (seed={seed})...")
    for game_number in range(num_games):
        weights = rng.integers(1, 20, size=num_agents)
        total = int(weights.sum())
        quota = int(rng.integers(max(1, int(total * 0.5)), int(total * 0.8) + 1))
        banzhaf, shapley = exact_power_indices(weights, quota)

        row = {f"weight_{i}": weights[i] for i in range(num_agents)}
        row["quota"] = quota
        row.update({f"banzhaf_target_{i}": banzhaf[i] for i in range(num_agents)})
        row.update({f"shapley_target_{i}": shapley[i] for i in range(num_agents)})
        rows.append(row)
        if num_games >= 10 and (game_number + 1) % (num_games // 10) == 0:
            print(f"  {game_number + 1}/{num_games}")

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path} | shape={dataframe.shape}")
    return dataframe


# Backward-compatible name used in earlier notebooks.
generate_midterm_2d_dataset = generate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate weighted games with exact Banzhaf and Shapley--Shubik labels."
    )
    parser.add_argument("--num-games", type=int, default=20_000)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/voting_games.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args.num_games, args.num_agents, args.seed, args.output)

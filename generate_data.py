"""Generate weighted voting or MCN games with power-index labels."""

import argparse
import os

import numpy as np
import pandas as pd

from src.banzhaf import exact_power_indices
from src.mcn import generate_dataset as generate_mcn_dataset


def generate_weighted_dataset(
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


def save_mcn_dataset(dataset, output_path):
    """Save an MCN dataset as a compressed NumPy archive."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(output_path, **dataset)
    print(
        "Dataset saved to "
        f"{output_path} | rules={dataset['rules'].shape} | "
        f"targets={dataset['banzhaf_targets'].shape}"
    )


# Backward-compatible public name used by earlier notebooks.
generate_dataset = generate_weighted_dataset


# Backward-compatible name used in earlier notebooks.
generate_midterm_2d_dataset = generate_weighted_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate weighted voting or MCN games with Banzhaf and Shapley labels."
    )
    parser.add_argument("--game-type", choices=("weighted", "mcn"), default="weighted")
    parser.add_argument("--num-games", type=int, default=20_000)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--num-rules", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--rule-generator",
        choices=("uniform", "coin_flip", "gaussian_mixture"),
        default="uniform",
        help="MCN rule-generation strategy from the paper.",
    )
    parser.add_argument(
        "--value-generator",
        choices=("uniform", "low_variance", "high_variance"),
        default="uniform",
        help="MCN rule-value generation strategy from the paper.",
    )
    parser.add_argument("--p", type=float, default=0.5, help="MCN threshold controlling rule density.")
    parser.add_argument("--num-coins", type=int, default=None, help="Coin-flip MCN assignments per rule.")
    parser.add_argument("--gamma-shape", type=float, default=2.0)
    parser.add_argument("--gamma-rate", type=float, default=2.0)
    parser.add_argument(
        "--label-method",
        choices=("exact", "monte_carlo"),
        default="exact",
        help="Use exact MCN labels for small games or Monte Carlo labels for larger games.",
    )
    parser.add_argument("--monte-carlo-samples", type=int, default=10_000)
    parser.add_argument(
        "--monte-carlo-batch-size",
        type=int,
        default=2_048,
        help="MCN Monte Carlo samples processed per NumPy batch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output
    if output is None:
        output = "data/voting_games.csv" if args.game_type == "weighted" else "data/mcn_games.npz"

    if args.game_type == "weighted":
        generate_weighted_dataset(args.num_games, args.num_agents, args.seed, output)
    else:
        print(
            "Generating "
            f"{args.num_games} MCN games with {args.num_agents} agents, "
            f"{args.num_rules} rules, {args.rule_generator} rules, "
            f"{args.value_generator} values (seed={args.seed})..."
        )
        dataset = generate_mcn_dataset(
            num_games=args.num_games,
            num_rules=args.num_rules,
            num_agents=args.num_agents,
            seed=args.seed,
            rule_generator=args.rule_generator,
            value_generator=args.value_generator,
            p=args.p,
            label_method=args.label_method,
            monte_carlo_samples=args.monte_carlo_samples,
            monte_carlo_batch_size=args.monte_carlo_batch_size,
            num_coins=args.num_coins,
            gamma_shape=args.gamma_shape,
            gamma_rate=args.gamma_rate,
        )
        save_mcn_dataset(dataset, output)

"""Compare Monte Carlo Banzhaf sampling methods and confidence intervals."""

import argparse
import os

import numpy as np
import pandas as pd

from src.banzhaf import exact_power_indices, monte_carlo_banzhaf
from src.plots import save_monte_carlo_interval_chart, save_power_index_comparison


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=float, default=[4, 2, 7, 1, 5, 3, 6, 2])
    parser.add_argument("--quota", type=float, default=16)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main(args=None):
    args = parse_args() if args is None else args
    weights = np.asarray(args.weights, dtype=float)
    exact, exact_shapley = exact_power_indices(weights, args.quota)
    results = {
        method.title(): monte_carlo_banzhaf(
            weights,
            args.quota,
            num_samples=args.samples,
            seed=args.seed,
            method=method,
            return_result=True,
        )
        for method in ("plain", "antithetic", "stratified")
    }
    rows = []
    for label, result in results.items():
        for agent in range(weights.size):
            rows.append(
                {
                    "method": label,
                    "agent": agent,
                    "exact": exact[agent],
                    "estimate": result.estimate[agent],
                    "standard_error": result.standard_error[agent],
                    "ci_low": result.ci_low[agent],
                    "ci_high": result.ci_high[agent],
                }
            )
    os.makedirs(args.results_dir, exist_ok=True)
    table_path = os.path.join(args.results_dir, "monte_carlo_confidence_intervals.csv")
    chart_path = os.path.join(args.results_dir, "monte_carlo_confidence_intervals.png")
    index_chart_path = os.path.join(args.results_dir, "exact_power_indices.png")
    pd.DataFrame(rows).to_csv(table_path, index=False)
    save_monte_carlo_interval_chart(exact, results, chart_path)
    save_power_index_comparison(exact, exact_shapley, index_chart_path)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nSaved {table_path}, {chart_path}, and {index_chart_path}")


if __name__ == "__main__":
    main()

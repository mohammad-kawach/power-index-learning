"""Feature engineering shared by training and prediction."""

import re

import numpy as np


def _numbered_columns(columns, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    found = [(int(match.group(1)), column) for column in columns if (match := pattern.match(column))]
    found.sort()
    if found and [number for number, _ in found] != list(range(len(found))):
        raise ValueError(f"{prefix} columns must be consecutively numbered from zero")
    return [column for _, column in found]


def infer_num_agents(df) -> int:
    """Infer agent count from consecutively numbered ``weight_*`` columns."""
    columns = _numbered_columns(df.columns, "weight_")
    if not columns:
        raise ValueError("dataset has no weight_0, weight_1, ... columns")
    return len(columns)


def target_columns(df, index_name):
    """Find ordered target columns, including old Banzhaf-only datasets."""
    columns = _numbered_columns(df.columns, f"{index_name}_target_")
    if not columns and index_name == "banzhaf":
        columns = _numbered_columns(df.columns, "target_")
    return columns


def create_features_for_one_game(weights: np.ndarray, quota: float) -> np.ndarray:
    """Generate ``3 * n_agents + 2`` scale-aware tabular features."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    total_weight = np.sum(weights)
    norm_weights = weights / (total_weight + 1e-8)
    quota_coverage = weights / (float(quota) + 1e-8)
    relative_quota = np.array([float(quota) / (total_weight + 1e-8)])
    return np.concatenate([weights, [quota], norm_weights, quota_coverage, relative_quota])


def create_feature_matrix(df):
    """Convert a weighted-game dataframe into a feature matrix."""
    num_agents = infer_num_agents(df)
    weights = df[[f"weight_{i}" for i in range(num_agents)]].values
    quotas = df["quota"].values
    return np.array(
        [create_features_for_one_game(row, quota) for row, quota in zip(weights, quotas)],
        dtype=float,
    )


def clean_prediction(prediction: np.ndarray) -> np.ndarray:
    """Project one model output onto the probability simplex."""
    prediction = np.asarray(prediction, dtype=float)
    prediction = np.maximum(prediction, 0)
    total = prediction.sum()
    return prediction / total if total > 0 else np.ones_like(prediction) / len(prediction)

"""Feature engineering shared by training and prediction."""

import re

import numpy as np

from src.mcn import validate_rule_batch, validate_rule_tensor


MCN_FEATURE_SETS = ("raw", "augmented")


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


def _validate_mcn_feature_set(feature_set: str) -> str:
    if feature_set not in MCN_FEATURE_SETS:
        raise ValueError("feature_set must be 'raw' or 'augmented'")
    return feature_set


def create_mcn_aggregate_features(rules: np.ndarray) -> np.ndarray:
    """Create MCN-aware summary features for one rule matrix.

    The flattened tensor preserves exact rule identity. These summaries add
    scale-aware signals that tabular regressors otherwise need to rediscover:
    how often each agent is required, banned, or mentioned, the value mass tied
    to each role, and global rule-complexity statistics.
    """
    rules = validate_rule_tensor(rules)
    num_rules = rules.shape[0]
    num_agents = (rules.shape[1] - 1) // 2
    required = rules[:, :num_agents]
    banned = rules[:, num_agents : 2 * num_agents]
    values = rules[:, -1]
    mentioned = required + banned
    total_value = values.sum()
    value_denominator = total_value + 1e-8

    required_counts = required.sum(axis=0) / num_rules
    banned_counts = banned.sum(axis=0) / num_rules
    mention_counts = mentioned.sum(axis=0) / num_rules
    required_value_share = (required * values[:, None]).sum(axis=0) / value_denominator
    banned_value_share = (banned * values[:, None]).sum(axis=0) / value_denominator
    role_value_share = required_value_share + banned_value_share
    signed_role_value_share = required_value_share - banned_value_share

    per_agent = np.concatenate(
        [
            required_counts,
            banned_counts,
            mention_counts,
            required_value_share,
            banned_value_share,
            role_value_share,
            signed_role_value_share,
        ]
    )

    required_per_rule = required.sum(axis=1) / num_agents
    banned_per_rule = banned.sum(axis=1) / num_agents
    mentioned_per_rule = mentioned.sum(axis=1) / num_agents
    normalized_values = values / value_denominator
    global_stats = np.array(
        [
            values.mean(),
            values.std(),
            values.min(),
            values.max(),
            total_value,
            required_per_rule.mean(),
            required_per_rule.std(),
            required_per_rule.max(),
            banned_per_rule.mean(),
            banned_per_rule.std(),
            banned_per_rule.max(),
            mentioned_per_rule.mean(),
            mentioned_per_rule.std(),
            mentioned_per_rule.max(),
            normalized_values.max(),
            normalized_values.std(),
        ],
        dtype=float,
    )
    return np.concatenate([per_agent, global_stats]).astype(float)


def create_features_for_one_mcn(rules: np.ndarray, feature_set: str = "augmented") -> np.ndarray:
    """Convert one MCN rule matrix into model-ready features.

    The project keeps the dataset itself as a real 3D tensor. The flattening is
    only the final handoff to feedforward regressors. ``feature_set="raw"``
    returns only the flattened tensor. ``feature_set="augmented"`` appends
    deterministic MCN-aware aggregate features.
    """
    feature_set = _validate_mcn_feature_set(feature_set)
    rules = validate_rule_tensor(rules)
    raw_features = rules.reshape(-1).astype(float)
    if feature_set == "raw":
        return raw_features
    return np.concatenate([raw_features, create_mcn_aggregate_features(rules)])


def create_mcn_feature_matrix(rule_tensor: np.ndarray, feature_set: str = "augmented") -> np.ndarray:
    """Convert a batch of MCN tensors into a 2D feature matrix for models."""
    feature_set = _validate_mcn_feature_set(feature_set)
    rule_tensor = validate_rule_batch(rule_tensor)
    return np.array(
        [create_features_for_one_mcn(rules, feature_set=feature_set) for rules in rule_tensor],
        dtype=float,
    )

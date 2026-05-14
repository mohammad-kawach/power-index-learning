import numpy as np

NUM_AGENTS = 5


def create_features(weights: np.ndarray, quota: int) -> np.ndarray:
    """
    Create simple engineered features for a voting game.

    This helps all models learn better than using only raw weights + quota.
    Features are still easy to explain:
    - raw weights
    - quota
    - total weight
    - weight / total weight
    - quota / total weight
    - weight / quota
    - distance from quota if this agent acts alone
    """
    weights = np.asarray(weights, dtype=float)
    quota = float(quota)
    total_weight = float(weights.sum())

    normalized_weights = weights / (total_weight + 1e-8)
    quota_ratio = np.array([quota / (total_weight + 1e-8)])
    weight_quota_ratio = weights / (quota + 1e-8)
    distance_to_quota = (quota - weights) / (total_weight + 1e-8)

    return np.concatenate([
        weights,
        np.array([quota, total_weight]),
        normalized_weights,
        quota_ratio,
        weight_quota_ratio,
        distance_to_quota,
    ])


def create_feature_matrix(df):
    weights_matrix = df[[f"weight_{i}" for i in range(NUM_AGENTS)]].values
    quotas = df["quota"].values
    return np.array([
        create_features(weights_matrix[i], quotas[i])
        for i in range(len(df))
    ], dtype=float)


def clean_prediction(prediction: np.ndarray) -> np.ndarray:
    """Make predictions valid Banzhaf-like distributions."""
    prediction = np.asarray(prediction, dtype=float)
    prediction = np.maximum(prediction, 0)
    total = prediction.sum()
    if total > 0:
        prediction = prediction / total
    else:
        prediction = np.ones_like(prediction) / len(prediction)
    return prediction

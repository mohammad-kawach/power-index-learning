import numpy as np

# Standardized for the 5-agent midterm (2D Tabular version)
NUM_AGENTS = 5

def create_features_for_one_game(weights: np.ndarray, quota: float) -> np.ndarray:
    """
    Generates 17 engineered features for 2D Weighted Voting Games.
    """
    weights = np.asarray(weights, dtype=float)
    total_weight = np.sum(weights)
    
    # 1. Raw Weights (5 features)
    # 2. Quota (1 feature)
    # 3. Normalized weights (5 features)
    norm_weights = weights / (total_weight + 1e-8)
    
    # 4. Quota coverage per agent (5 features)
    quota_coverage = weights / (quota + 1e-8)
    
    # 5. Quota as percentage of total weight (1 feature)
    relative_quota = np.array([quota / (total_weight + 1e-8)])
    
    # Total: 5 + 1 + 5 + 5 + 1 = 17 features
    return np.concatenate([
        weights, 
        [quota], 
        norm_weights, 
        quota_coverage, 
        relative_quota
    ])

def create_feature_matrix(df):
    """Processes the entire 2D Tabular dataframe into a feature matrix."""
    weights_matrix = df[[f"weight_{i}" for i in range(NUM_AGENTS)]].values
    quotas = df["quota"].values
    return np.array([
        create_features_for_one_game(weights_matrix[i], quotas[i])
        for i in range(len(df))
    ], dtype=float)

def clean_prediction(prediction: np.ndarray) -> np.ndarray:
    """Ensures outputs are valid power index distributions[cite: 77, 149]."""
    prediction = np.asarray(prediction, dtype=float)
    prediction = np.maximum(prediction, 0)
    total = prediction.sum()
    return prediction / total if total > 0 else np.ones_like(prediction) / len(prediction)
import numpy as np


class DecisionTreeRegressorScratch:
    """Small multi-output regression tree from scratch."""

    def __init__(self, max_depth=8, min_samples_split=10, max_features=None, extra_random=False, seed=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.extra_random = extra_random
        self.rng = np.random.default_rng(seed)
        self.tree = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _mse(self, y):
        if len(y) == 0:
            return 0.0
        return float(np.mean(np.sum((y - y.mean(axis=0)) ** 2, axis=1)))

    def _feature_indices(self):
        if self.max_features is None:
            k = self.n_features_
        elif isinstance(self.max_features, str) and self.max_features == "sqrt":
            k = max(1, int(np.sqrt(self.n_features_)))
        else:
            k = int(self.max_features)
        return self.rng.choice(self.n_features_, size=k, replace=False)

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_score = float("inf")
        parent_n = len(y)

        for feature in self._feature_indices():
            values = X[:, feature]
            unique_values = np.unique(values)
            if len(unique_values) <= 1:
                continue

            if self.extra_random:
                thresholds = [self.rng.uniform(values.min(), values.max())]
            else:
                # Use candidate midpoints. Limit candidates for speed.
                candidates = (unique_values[:-1] + unique_values[1:]) / 2
                if len(candidates) > 20:
                    candidates = self.rng.choice(candidates, size=20, replace=False)
                thresholds = candidates

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]
                score = (len(left_y) / parent_n) * self._mse(left_y) + (len(right_y) / parent_n) * self._mse(right_y)

                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        value = y.mean(axis=0)

        if depth >= self.max_depth or len(X) < self.min_samples_split * 2:
            return {"value": value}

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {"value": value}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            "feature": feature,
            "threshold": threshold,
            "value": value,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _predict_one(self, x, node):
        if "left" not in node:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class ForestRegressorScratch:
    """
    From-scratch Random Forest / Extra Trees style multi-output regressor.

    mode="random_forest": bootstrap samples + best random feature splits
    mode="extra_trees": no bootstrap + random thresholds for more randomness
    """

    def __init__(self, n_estimators=40, max_depth=9, min_samples_split=8, max_features="sqrt", mode="random_forest", seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.mode = mode
        self.seed = seed
        self.trees = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.trees = []
        n = len(X)

        for i in range(self.n_estimators):
            if self.mode == "random_forest":
                indices = rng.integers(0, n, size=n)
                X_train = X[indices]
                y_train = y[indices]
                extra_random = False
            elif self.mode == "extra_trees":
                X_train = X
                y_train = y
                extra_random = True
            else:
                raise ValueError("mode must be 'random_forest' or 'extra_trees'")

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                extra_random=extra_random,
                seed=int(rng.integers(0, 1_000_000)),
            )
            tree.fit(X_train, y_train)
            self.trees.append(tree)
            print(f"Trained {self.mode} tree {i + 1}/{self.n_estimators}")
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return predictions.mean(axis=0)

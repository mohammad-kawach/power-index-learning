import numpy as np


class NumpyMLP:
    """
    A small neural network implemented from scratch with NumPy.

    Architecture:
    input -> Dense -> ReLU -> Dense -> ReLU -> Dense -> Softmax

    Softmax is used because normalized Banzhaf values are a distribution:
    all values are positive and sum to 1.
    """

    def __init__(self, input_size, hidden1=128, hidden2=64, output_size=5, learning_rate=0.001, seed=42):
        rng = np.random.default_rng(seed)
        self.learning_rate = learning_rate

        self.W1 = rng.normal(0, np.sqrt(2 / input_size), size=(input_size, hidden1))
        self.b1 = np.zeros((1, hidden1))
        self.W2 = rng.normal(0, np.sqrt(2 / hidden1), size=(hidden1, hidden2))
        self.b2 = np.zeros((1, hidden2))
        self.W3 = rng.normal(0, np.sqrt(2 / hidden2), size=(hidden2, output_size))
        self.b3 = np.zeros((1, output_size))

        # Adam optimizer state
        self.t = 0
        self.m = {}
        self.v = {}
        for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            self.m[name] = np.zeros_like(getattr(self, name))
            self.v[name] = np.zeros_like(getattr(self, name))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)
        z3 = a2 @ self.W3 + self.b3
        predictions = self.softmax(z3)
        cache = (X, z1, a1, z2, a2, z3, predictions)
        return predictions, cache

    def loss(self, predictions, y):
        return -np.mean(np.sum(y * np.log(predictions + 1e-8), axis=1))

    def _adam_update(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for name, grad in grads.items():
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (grad ** 2)

            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)

            value = getattr(self, name)
            value -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, value)

    def train(self, X_train, y_train, X_test, y_test, epochs=800, batch_size=256, print_every=50):
        history = []
        n = len(X_train)
        rng = np.random.default_rng(123)

        for epoch in range(1, epochs + 1):
            indices = rng.permutation(n)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                m = len(X_batch)

                predictions, cache = self.forward(X_batch)
                X, z1, a1, z2, a2, z3, _ = cache

                dz3 = (predictions - y_batch) / m
                dW3 = a2.T @ dz3
                db3 = np.sum(dz3, axis=0, keepdims=True)

                da2 = dz3 @ self.W3.T
                dz2 = da2 * self.relu_derivative(z2)
                dW2 = a1.T @ dz2
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * self.relu_derivative(z1)
                dW1 = X.T @ dz1
                db1 = np.sum(dz1, axis=0, keepdims=True)

                grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
                self._adam_update(grads)

            train_pred, _ = self.forward(X_train)
            test_pred, _ = self.forward(X_test)
            train_loss = self.loss(train_pred, y_train)
            test_mae = np.mean(np.abs(test_pred - y_test))
            history.append((epoch, train_loss, test_mae))

            if epoch == 1 or epoch % print_every == 0:
                print(f"Epoch {epoch:4d} | Loss: {train_loss:.4f} | Test MAE: {test_mae:.4f}")

        return np.array(history)

    def predict(self, X):
        predictions, _ = self.forward(X)
        return predictions

    def save(self, path, scaler):
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            mean=scaler.mean_, std=scaler.std_,
            learning_rate=self.learning_rate,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(
            input_size=data["W1"].shape[0],
            hidden1=data["W1"].shape[1],
            hidden2=data["W2"].shape[1],
            output_size=data["W3"].shape[1],
            learning_rate=float(data["learning_rate"]),
        )
        for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            setattr(model, name, data[name])
        mean = data["mean"]
        std = data["std"]
        return model, mean, std

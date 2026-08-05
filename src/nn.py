import numpy as np


class NumpyMLP:
    """
    A small fully connected neural network implemented from scratch.

    Weighted voting games use the original softmax classifier-style output.
    MCN games use the paper-aligned regression setup:
    input -> 512 -> 256 -> 128 -> linear output, trained with MSE.
    """

    def __init__(
        self,
        input_size,
        hidden1=128,
        hidden2=64,
        output_size=5,
        learning_rate=0.001,
        seed=42,
        *,
        hidden_layers=None,
        output_activation="softmax",
        loss="cross_entropy",
        dropout_rate=0.0,
    ):
        if hidden_layers is None:
            hidden_layers = (hidden1, hidden2)
        if output_activation not in {"softmax", "linear"}:
            raise ValueError("output_activation must be 'softmax' or 'linear'")
        if loss not in {"cross_entropy", "mse"}:
            raise ValueError("loss must be 'cross_entropy' or 'mse'")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1)")

        self.learning_rate = learning_rate
        self.output_activation = output_activation
        self.loss_name = loss
        self.dropout_rate = dropout_rate
        self.hidden_layers = tuple(int(width) for width in hidden_layers)
        layer_sizes = (int(input_size), *self.hidden_layers, int(output_size))
        if any(width <= 0 for width in layer_sizes):
            raise ValueError("all layer sizes must be positive")

        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(rng.normal(0, np.sqrt(2 / fan_in), size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

        self.t = 0
        self.m_weights = [np.zeros_like(weight) for weight in self.weights]
        self.v_weights = [np.zeros_like(weight) for weight in self.weights]
        self.m_biases = [np.zeros_like(bias) for bias in self.biases]
        self.v_biases = [np.zeros_like(bias) for bias in self.biases]

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

    def forward(self, X, *, training=False, rng=None):
        activation = X
        hidden_cache = []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = activation @ weight + bias
            next_activation = self.relu(z)
            dropout_mask = None
            if training and self.dropout_rate > 0:
                if rng is None:
                    raise ValueError("rng is required when training with dropout")
                dropout_mask = (rng.random(next_activation.shape) >= self.dropout_rate).astype(float)
                dropout_mask /= 1 - self.dropout_rate
                next_activation *= dropout_mask
            hidden_cache.append((activation, z, dropout_mask))
            activation = next_activation

        logits = activation @ self.weights[-1] + self.biases[-1]
        if self.output_activation == "softmax":
            predictions = self.softmax(logits)
        else:
            predictions = logits
        cache = (hidden_cache, activation, logits, predictions)
        return predictions, cache

    def loss(self, predictions, y):
        if self.loss_name == "cross_entropy":
            return -float(np.mean(np.sum(y * np.log(predictions + 1e-8), axis=1)))
        return float(np.mean((predictions - y) ** 2))

    def _adam_update(self, grad_weights, grad_biases, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for index, (grad_w, grad_b) in enumerate(zip(grad_weights, grad_biases)):
            self.m_weights[index] = beta1 * self.m_weights[index] + (1 - beta1) * grad_w
            self.v_weights[index] = beta2 * self.v_weights[index] + (1 - beta2) * (grad_w**2)
            self.m_biases[index] = beta1 * self.m_biases[index] + (1 - beta1) * grad_b
            self.v_biases[index] = beta2 * self.v_biases[index] + (1 - beta2) * (grad_b**2)

            m_w_hat = self.m_weights[index] / (1 - beta1**self.t)
            v_w_hat = self.v_weights[index] / (1 - beta2**self.t)
            m_b_hat = self.m_biases[index] / (1 - beta1**self.t)
            v_b_hat = self.v_biases[index] / (1 - beta2**self.t)

            self.weights[index] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.biases[index] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def _output_gradient(self, predictions, y):
        batch_size = len(y)
        if self.output_activation == "softmax" and self.loss_name == "cross_entropy":
            return (predictions - y) / batch_size
        if self.output_activation == "linear" and self.loss_name == "mse":
            return 2 * (predictions - y) / predictions.size
        raise ValueError("unsupported output_activation/loss combination")

    def train(
        self,
        X_train,
        y_train,
        X_eval,
        y_eval,
        epochs=800,
        batch_size=256,
        print_every=50,
        evaluation_label="Evaluation MAE",
    ):
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

                predictions, cache = self.forward(X_batch, training=True, rng=rng)
                hidden_cache, last_hidden, _, _ = cache
                delta = self._output_gradient(predictions, y_batch)

                activations = [entry[0] for entry in hidden_cache] + [last_hidden]
                grad_weights = [None] * len(self.weights)
                grad_biases = [None] * len(self.biases)
                grad_weights[-1] = activations[-1].T @ delta
                grad_biases[-1] = np.sum(delta, axis=0, keepdims=True)

                upstream = delta @ self.weights[-1].T
                for layer_index in range(len(self.weights) - 2, -1, -1):
                    previous_activation, z, dropout_mask = hidden_cache[layer_index]
                    if dropout_mask is not None:
                        upstream *= dropout_mask
                    delta_hidden = upstream * self.relu_derivative(z)
                    grad_weights[layer_index] = previous_activation.T @ delta_hidden
                    grad_biases[layer_index] = np.sum(delta_hidden, axis=0, keepdims=True)
                    if layer_index > 0:
                        upstream = delta_hidden @ self.weights[layer_index].T

                self._adam_update(grad_weights, grad_biases)

            train_pred, _ = self.forward(X_train)
            eval_pred, _ = self.forward(X_eval)
            train_loss = self.loss(train_pred, y_train)
            eval_mae = np.mean(np.abs(eval_pred - y_eval))
            history.append((epoch, train_loss, eval_mae))

            if epoch == 1 or epoch % print_every == 0:
                print(
                    f"Epoch {epoch:4d} | Loss: {train_loss:.4f} | "
                    f"{evaluation_label}: {eval_mae:.4f}"
                )

        return np.array(history)

    def predict(self, X):
        predictions, _ = self.forward(X)
        return predictions

    def save(self, path, scaler, metadata=None):
        arrays = {
            "num_layers": np.array([len(self.weights)]),
            "hidden_layers": np.asarray(self.hidden_layers, dtype=int),
            "output_activation": np.array([self.output_activation]),
            "loss_name": np.array([self.loss_name]),
            "dropout_rate": np.array([self.dropout_rate], dtype=float),
            "learning_rate": np.array([self.learning_rate], dtype=float),
            "mean": scaler.mean_,
            "std": scaler.std_,
        }
        for key, value in (metadata or {}).items():
            arrays[f"metadata_{key}"] = np.array([str(value)])
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{index}"] = weight
            arrays[f"b{index}"] = bias
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        if "num_layers" in data:
            num_layers = int(data["num_layers"][0])
            hidden_layers = tuple(int(value) for value in data["hidden_layers"])
            model = cls(
                input_size=data["W0"].shape[0],
                hidden_layers=hidden_layers,
                output_size=data[f"W{num_layers - 1}"].shape[1],
                learning_rate=float(data["learning_rate"][0]),
                output_activation=str(data["output_activation"][0]),
                loss=str(data["loss_name"][0]),
                dropout_rate=float(data["dropout_rate"][0]),
            )
            model.weights = [data[f"W{index}"] for index in range(num_layers)]
            model.biases = [data[f"b{index}"] for index in range(num_layers)]
        else:
            # Backward compatibility for model files produced by the original
            # two-hidden-layer implementation.
            model = cls(
                input_size=data["W1"].shape[0],
                hidden1=data["W1"].shape[1],
                hidden2=data["W2"].shape[1],
                output_size=data["W3"].shape[1],
                learning_rate=float(data["learning_rate"]),
            )
            model.weights = [data["W1"], data["W2"], data["W3"]]
            model.biases = [data["b1"], data["b2"], data["b3"]]

        model.m_weights = [np.zeros_like(weight) for weight in model.weights]
        model.v_weights = [np.zeros_like(weight) for weight in model.weights]
        model.m_biases = [np.zeros_like(bias) for bias in model.biases]
        model.v_biases = [np.zeros_like(bias) for bias in model.biases]
        mean = data["mean"]
        std = data["std"]
        return model, mean, std

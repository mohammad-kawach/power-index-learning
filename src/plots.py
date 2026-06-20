import os
import tempfile

# Matplotlib may try to write under ~/.config, which is not always writable.
if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache_dir = os.path.join(tempfile.gettempdir(), "matplotlib-cache")
    os.makedirs(matplotlib_cache_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = matplotlib_cache_dir

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_output_dir(output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def save_prediction_chart(real, predictions_dict, output_path):
    _ensure_output_dir(output_path)
    agents = [f"Agent {i}" for i in range(len(real))]
    x = np.arange(len(real))
    labels = ["Real"] + list(predictions_dict.keys())
    all_values = [real] + list(predictions_dict.values())
    width = 0.8 / len(all_values)

    plt.figure(figsize=(10, 5))
    for i, values in enumerate(all_values):
        offset = (i - (len(all_values) - 1) / 2) * width
        plt.bar(x + offset, values, width, label=labels[i])

    plt.xlabel("Agents")
    plt.ylabel("Banzhaf Power")
    plt.title("Real vs Predicted Banzhaf Values")
    plt.xticks(x, agents)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_loss_curve(history, output_path):
    _ensure_output_dir(output_path)
    epochs = history[:, 0]

    fig, loss_axis = plt.subplots(figsize=(9, 5))
    mae_axis = loss_axis.twinx()

    loss_line = loss_axis.plot(epochs, history[:, 1], color="tab:blue", label="Training Loss")
    mae_line = mae_axis.plot(epochs, history[:, 2], color="tab:orange", label="Test MAE")

    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Training Loss", color="tab:blue")
    mae_axis.set_ylabel("Test MAE", color="tab:orange")
    loss_axis.tick_params(axis="y", labelcolor="tab:blue")
    mae_axis.tick_params(axis="y", labelcolor="tab:orange")
    loss_axis.set_title("Neural Network Training Curve")
    loss_axis.grid(True, axis="x", alpha=0.25)

    lines = loss_line + mae_line
    labels = [line.get_label() for line in lines]
    loss_axis.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_model_mae_chart(metrics, output_path):
    _ensure_output_dir(output_path)
    model_names = list(metrics["model"])
    mae_values = np.asarray(metrics["test_mae"], dtype=float)

    fig, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.barh(model_names, mae_values, color=["tab:blue", "tab:green", "tab:orange"])
    axis.invert_yaxis()
    axis.set_xlabel("Test MAE (lower is better)")
    axis.set_title("Model Mean Absolute Error Comparison")
    axis.grid(True, axis="x", alpha=0.25)

    max_mae = max(mae_values.max(), 1e-8)
    axis.set_xlim(0, max_mae * 1.2)
    for bar, value in zip(bars, mae_values):
        axis.text(
            value + max_mae * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_prediction_scatter(real, predictions_dict, output_path):
    _ensure_output_dir(output_path)
    real_values = np.asarray(real, dtype=float).ravel()

    fig, axis = plt.subplots(figsize=(7, 7))
    max_value = real_values.max()

    for label, predictions in predictions_dict.items():
        predicted_values = np.asarray(predictions, dtype=float).ravel()
        max_value = max(max_value, predicted_values.max())
        axis.scatter(real_values, predicted_values, s=12, alpha=0.35, label=label)

    axis_limit = max(max_value * 1.08, 0.05)
    axis.plot([0, axis_limit], [0, axis_limit], color="black", linestyle="--", linewidth=1, label="Perfect prediction")
    axis.set_xlim(0, axis_limit)
    axis.set_ylim(0, axis_limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Exact Banzhaf Value")
    axis.set_ylabel("Predicted Banzhaf Value")
    axis.set_title("Predicted vs Exact Banzhaf Values")
    axis.grid(True, alpha=0.25)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_per_agent_error_chart(
    real,
    predictions_dict,
    output_path,
    title="Per-Agent Prediction Error",
    ylabel="Mean Absolute Error",
):
    _ensure_output_dir(output_path)
    real_values = np.asarray(real, dtype=float)
    agents = [f"Agent {i}" for i in range(real_values.shape[-1])]
    x = np.arange(len(agents))
    width = 0.8 / len(predictions_dict)

    fig, axis = plt.subplots(figsize=(10, 5))
    for i, (label, predictions) in enumerate(predictions_dict.items()):
        predicted_values = np.asarray(predictions, dtype=float)
        absolute_errors = np.abs(predicted_values - real_values)
        if absolute_errors.ndim == 2:
            absolute_errors = absolute_errors.mean(axis=0)

        offset = (i - (len(predictions_dict) - 1) / 2) * width
        axis.bar(x + offset, absolute_errors, width, label=label)

    axis.set_xlabel("Agents")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xticks(x)
    axis.set_xticklabels(agents)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

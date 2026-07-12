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


def save_prediction_chart(
    real,
    predictions_dict,
    output_path,
    title="Real vs Predicted Banzhaf Values",
    ylabel="Banzhaf Power",
):
    _ensure_output_dir(output_path)
    agents = [f"Agent {i}" for i in range(len(real))]
    x = np.arange(len(real))
    labels = ["Real"] + list(predictions_dict.keys())
    all_values = [real] + list(predictions_dict.values())
    width = 0.8 / len(all_values)

    plt.figure(figsize=(max(10, len(real) * 1.25), 5.5))
    for i, values in enumerate(all_values):
        offset = (i - (len(all_values) - 1) / 2) * width
        plt.bar(x + offset, values, width, label=labels[i])

    plt.xlabel("Agents")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, agents)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_loss_curve(history, output_path, title="Neural Network Training Curve"):
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
    loss_axis.set_title(title)
    loss_axis.grid(True, axis="x", alpha=0.25)

    lines = loss_line + mae_line
    labels = [line.get_label() for line in lines]
    loss_axis.legend(lines, labels, loc="best")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_model_mae_chart(metrics, output_path):
    _ensure_output_dir(output_path)
    if "index" in metrics:
        model_names = [f"{index}: {model}" for index, model in zip(metrics["index"], metrics["model"])]
    else:
        model_names = list(metrics["model"])
    mae_values = np.asarray(metrics["test_mae"], dtype=float)

    fig_height = max(4.8, len(model_names) * 0.43)
    fig, axis = plt.subplots(figsize=(11, fig_height))
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))
    bars = axis.barh(model_names, mae_values, color=colors)
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


def save_prediction_scatter(
    real,
    predictions_dict,
    output_path,
    title="Predicted vs Exact Values",
    xlabel="Exact value",
    ylabel="Predicted value",
):
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
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
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


def save_monte_carlo_interval_chart(exact, results, output_path):
    """Plot Monte Carlo point estimates and confidence intervals by method."""
    _ensure_output_dir(output_path)
    exact = np.asarray(exact, dtype=float)
    agents = np.arange(exact.size)
    fig, axis = plt.subplots(figsize=(max(10, exact.size * 1.2), 5.5))
    offsets = np.linspace(-0.22, 0.22, len(results))
    for offset, (label, result) in zip(offsets, results.items()):
        errors = np.vstack((result.estimate - result.ci_low, result.ci_high - result.estimate))
        axis.errorbar(
            agents + offset,
            result.estimate,
            yerr=errors,
            fmt="o",
            capsize=3,
            label=label,
        )
    axis.scatter(agents, exact, marker="x", s=65, color="black", label="Exact")
    axis.set_xticks(agents)
    axis.set_xticklabels([f"Agent {i}" for i in agents])
    axis.set_xlabel("Agent")
    axis.set_ylabel("Normalized Banzhaf power")
    axis.set_title("Monte Carlo Banzhaf Estimates with 95% Confidence Intervals")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_power_index_comparison(banzhaf, shapley, output_path):
    """Plot exact Banzhaf and Shapley--Shubik values for one game."""
    _ensure_output_dir(output_path)
    banzhaf = np.asarray(banzhaf, dtype=float)
    shapley = np.asarray(shapley, dtype=float)
    agents = np.arange(banzhaf.size)
    width = 0.36
    fig, axis = plt.subplots(figsize=(max(10, banzhaf.size * 1.2), 5.2))
    axis.bar(agents - width / 2, banzhaf, width, label="Banzhaf")
    axis.bar(agents + width / 2, shapley, width, label="Shapley--Shubik")
    axis.set_xticks(agents)
    axis.set_xticklabels([f"Agent {i}" for i in agents])
    axis.set_xlabel("Agent")
    axis.set_ylabel("Normalized power")
    axis.set_title("Exact Power Indices for the Example Voting Game")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_mcn_rule_heatmap(rules, output_path):
    """Visualize one MCN rule matrix as required, banned, and absent agents."""
    _ensure_output_dir(output_path)
    rules = np.asarray(rules, dtype=float)
    num_agents = (rules.shape[1] - 1) // 2
    required = rules[:, :num_agents]
    banned = rules[:, num_agents : 2 * num_agents]
    values = rules[:, -1]
    membership = required - banned

    fig, (matrix_axis, value_axis) = plt.subplots(
        1,
        2,
        figsize=(max(9, num_agents * 0.8), max(4.5, rules.shape[0] * 0.22)),
        gridspec_kw={"width_ratios": [4, 1.3]},
    )
    image = matrix_axis.imshow(membership, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    matrix_axis.set_title("MCN Rule Membership")
    matrix_axis.set_xlabel("Agent")
    matrix_axis.set_ylabel("Rule")
    matrix_axis.set_xticks(np.arange(num_agents))
    matrix_axis.set_xticklabels([f"A{i}" for i in range(num_agents)])
    matrix_axis.set_yticks(np.arange(rules.shape[0]))
    matrix_axis.set_yticklabels([f"R{i}" for i in range(rules.shape[0])])
    colorbar = fig.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.04)
    colorbar.set_ticks([-1, 0, 1])
    colorbar.set_ticklabels(["banned", "absent", "required"])

    value_axis.barh(np.arange(rules.shape[0]), values, color="tab:green")
    value_axis.invert_yaxis()
    value_axis.set_title("Value")
    value_axis.set_xlabel("Weight")
    value_axis.set_yticks([])
    value_axis.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

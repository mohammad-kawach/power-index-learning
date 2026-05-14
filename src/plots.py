import os
import numpy as np
import matplotlib.pyplot as plt


def save_prediction_chart(real, predictions_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(history[:, 0], history[:, 1], label="Training Loss")
    plt.plot(history[:, 0], history[:, 2], label="Test MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Neural Network Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

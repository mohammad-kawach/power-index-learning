# InfluenceNet Mini — From-Scratch Banzhaf Prediction

This project predicts **Banzhaf power indices** for small weighted voting games.

The important point: the models are implemented **from scratch**, using only:

- Python
- NumPy
- Pandas
- Matplotlib

No scikit-learn is used.

## Idea

A weighted voting game has:

- several agents / voters
- one weight for each agent
- one quota

A coalition wins if the sum of its weights is at least the quota.

The **Banzhaf index** measures how powerful each agent is. An agent is powerful if adding it to a losing coalition often changes the result to winning.

Exact Banzhaf calculation is possible for 5 agents, but it becomes expensive for larger games. The goal of this mini project is to train machine learning models that approximate the exact Banzhaf values.

## Models

This project compares three from-scratch models:

| Model | Explanation |
|---|---|
| NumPy Neural Network | Two hidden layers, ReLU, softmax output, mini-batch training, Adam optimizer |
| From-Scratch Random Forest | Many regression trees trained on bootstrap samples |
| From-Scratch Extra Trees | Many randomized regression trees with random thresholds |

## Folder Structure

```text
influencenet_from_scratch/
├── data/
│   └── dataset.csv                 # created by generate_data.py
├── models/
│   ├── mlp_numpy.npz               # saved neural network
│   ├── random_forest_scratch.pkl    # saved random forest
│   └── extra_trees_scratch.pkl      # saved extra trees
├── results/
│   ├── mlp_training_curve.png
│   ├── model_metrics.csv
│   ├── example_prediction_table.csv
│   └── example_prediction_comparison.png
├── src/
│   ├── banzhaf.py
│   ├── features.py
│   ├── nn.py
│   ├── plots.py
│   ├── scaler.py
│   └── trees.py
├── generate_data.py
├── train_models.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

## Run the Project

### 1. Generate the dataset

```bash
python generate_data.py
```

This creates:

```text
data/dataset.csv
```

### 2. Train the models

```bash
python train_models.py
```

This trains:

- NumPy Neural Network
- From-Scratch Random Forest
- From-Scratch Extra Trees

It saves the trained models in:

```text
models/
```

It also saves metrics and the neural network training curve in:

```text
results/
```

### 3. Run one example prediction

```bash
python predict.py
```

The example uses:

```python
example_weights = np.array([4, 2, 7, 1, 5])
example_quota = 10
```

The script prints:

- real exact Banzhaf values
- neural network prediction
- random forest prediction
- extra trees prediction
- mean absolute error
- prediction time

It also saves a comparison chart:

```text
results/example_prediction_comparison.png
```

## What to Say in a Presentation

### Simple explanation

> In this project, I use machine learning to approximate Banzhaf power indices. The exact calculation checks all possible coalitions, which becomes expensive when the number of agents grows. So I generate many small voting games, calculate the exact Banzhaf values, and train models to predict them from the weights and quota.

### What changed from the old version?

> The old version only used one simple neural network prediction. In the new version, I compare three models: a NumPy neural network, a from-scratch Random Forest, and a from-scratch Extra Trees model. I also added engineered features, evaluation tables, prediction time comparison, and plots.

### Why softmax in the neural network?

> The Banzhaf values are normalized, so they are positive and sum to 1. Softmax makes the neural network output look like this kind of distribution.

### Why Random Forest and Extra Trees?

> They are good baselines. They use many decision trees and average their predictions. Random Forest uses bootstrap samples, while Extra Trees adds more randomness by choosing random split thresholds.

## Notes

The tree models are educational implementations. They are not as optimized as scikit-learn, but they are useful because the logic is visible and easy to explain.

For better accuracy, increase:

- `NUM_SAMPLES` in `generate_data.py`
- `epochs` in `train_models.py`
- `n_estimators` in `train_models.py`

For faster training, reduce those values.

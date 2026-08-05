# Python API

The command-line scripts cover the standard workflow. The examples below show
the main MCN functions for notebooks and custom experiments.

## Evaluate one MCN

```python
import numpy as np

from src.mcn import coalition_value, exact_power_indices

rules = np.array([
    [1, 1, 0, 0, 0, 0, 3],
    [1, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 1, 2],
], dtype=float)

print(coalition_value([1, 0, 0], rules))  # {a} -> 1
print(coalition_value([0, 1, 0], rules))  # {b} -> 2
print(coalition_value([1, 1, 0], rules))  # {a, b} -> 6

result = exact_power_indices(rules)
print(result.banzhaf)
print(result.shapley)
print(result.raw_banzhaf)
print(result.raw_shapley)
```

`result.banzhaf` and `result.shapley` are normalized by default. The `raw_*`
members retain the values before normalization.

## Generate one random rule matrix

```python
import numpy as np

from src.mcn import generate_random_rules

rng = np.random.default_rng(42)
rules = generate_random_rules(
    num_rules=20,
    num_agents=8,
    rng=rng,
    rule_generator="uniform",
    value_generator="low_variance",
    p=0.5,
)

print(rules.shape)  # (20, 17)
```

## Approximate labels with Monte Carlo

```python
from src.mcn import monte_carlo_power_indices

result = monte_carlo_power_indices(
    rules,
    num_samples=10000,
    seed=42,
    batch_size=2048,
)

print(result.banzhaf)
print(result.shapley)
```

## Build MCN model features

```python
from src.features import create_features_for_one_mcn

raw = create_features_for_one_mcn(rules, feature_set="raw")
augmented = create_features_for_one_mcn(rules, feature_set="augmented")

print(raw.shape)
print(augmented.shape)
```

`raw` is the flattened rule tensor. `augmented` keeps that tensor and appends
per-agent role/value summaries plus global rule-complexity statistics. The
training CLI defaults to `augmented`; pass `--mcn-feature-set raw` to reproduce
older raw-input benchmarks.

## Use signed contributions

The project defaults to normalized absolute changes. Standard signed marginal
contributions are available through keyword arguments:

```python
signed = exact_power_indices(
    rules,
    absolute=False,
    normalize=False,
)
```

## Project structure

| Path | Responsibility |
| --- | --- |
| `src/mcn.py` | MCN tensors, coalition values, labels, sampling, and generation |
| `src/banzhaf.py` | Weighted-voting exact indices and Monte Carlo Banzhaf |
| `src/features.py` | Weighted features, MCN tensor flattening, and MCN aggregate features |
| `src/nn.py` | From-scratch NumPy MLP |
| `src/trees.py` | From-scratch Random Forest and Extra Trees regressors |
| `src/scaler.py` | From-scratch standard scaling |
| `src/plots.py` | Headless Matplotlib charts |
| `generate_data.py` | Dataset-generation CLI |
| `train_models.py` | Training, evaluation, metrics, and plots |
| `predict.py` | Exact-versus-predicted example comparisons |
| `monte_carlo_demo.py` | Weighted-game sampling demo |
| `tests/` | Unit tests for weighted games, MCNs, features, and sampling |

Run any command with `--help` for its complete argument list:

```bash
python generate_data.py --help
python train_models.py --help
python predict.py --help
python monte_carlo_demo.py --help
```

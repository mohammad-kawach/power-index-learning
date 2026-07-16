# Experiments

This page records the canonical benchmark, alternative runs, generated
artifacts, and reproducibility limits.

## Canonical presentation benchmark

The repository's reference run is intentionally small enough to reproduce on a
laptop before a presentation.

| Setting | Value |
| --- | --- |
| Games / train / test | 500 / 400 / 100 |
| Agents / rules / input features | 6 / 12 / 156 |
| Rule/value generators | `uniform` / `uniform` |
| Labels | exact, absolute marginal changes, normalized per game |
| Monte Carlo samples | 0 used; the stored CLI default of 10,000 is ignored for exact labels |
| Split and model seed | 42 |
| NumPy MLP | 512-256-128, 20 epochs, batch size 256, dropout 0.2 |
| scikit-learn MLP | 512-256-128, at most 50 iterations, early stopping |
| Ensembles | 5 estimators each |
| Metric | element-wise test mean absolute error (MAE) |

From a fresh checkout and activated environment, run:

```bash
python generate_data.py \
  --game-type mcn \
  --num-games 500 \
  --num-agents 6 \
  --num-rules 12 \
  --rule-generator uniform \
  --value-generator uniform \
  --label-method exact \
  --seed 42 \
  --output data/mcn_games.npz

python train_models.py \
  --data data/mcn_games.npz \
  --game-type mcn \
  --epochs 20 \
  --sklearn-max-iter 50 \
  --n-estimators 5

python predict.py \
  --game-type mcn \
  --data data/mcn_games.npz \
  --example-index 0
```

### Numerical results

These values were reproduced on 16 July 2026. Lower is better.

| Model | Implementation | Banzhaf MAE | Shapley-Shubik MAE |
| --- | --- | ---: | ---: |
| Random Forest | from scratch | **0.039821** | 0.048048 |
| Extra Trees | from scratch | 0.039865 | **0.046667** |
| Random Forest | scikit-learn | 0.043160 | 0.051812 |
| Extra Trees | scikit-learn | 0.053382 | 0.062097 |
| MLP | scikit-learn | 0.070466 | 0.073790 |
| MLP | from scratch | 0.137822 | 0.136022 |

This is a deterministic educational benchmark on one small synthetic dataset,
not a reproduction of the paper's full experimental protocol and not evidence
of general state-of-the-art performance.

### Environment and runtime

| Component | Recorded value |
| --- | --- |
| Hardware | 14-core Apple M3 Max, 36 GB memory |
| Operating system | macOS 26.3.1, arm64 |
| Python | 3.9.6 |
| NumPy | 2.0.2 |
| Pandas | 2.3.3 |
| Matplotlib | 3.9.4 |
| scikit-learn | 1.6.1 |
| Dataset generation | 1.76 seconds wall time |
| Training, evaluation, and plots | 20.24 seconds wall time |
| Example prediction and plots | 2.08 seconds wall time |
| Total | 24.08 seconds wall time |

Timing used `/usr/bin/time -p` after package installation. Runtime varies with
hardware, background load, filesystem caching, and thread scheduling.

## Reading the figures

### Dataset structure

![Example MCN rule heatmap](../results/example_mcn_rules.png)

Blue cells are required agents, red cells are banned agents, neutral cells are
unused agents, and the side bars are rule values.

![MCN rule complexity](../results/mcn_rule_complexity.png)

This summarizes how many agents are required, banned, or mentioned per rule and
shows the rule-value distribution.

![MCN target distribution](../results/mcn_target_distribution.png)

This checks the distribution of normalized targets and their per-agent means
and standard deviations.

![Banzhaf and Shapley relationship](../results/mcn_index_relationship.png)

This compares the two exact target definitions for the same agent-game pairs.

### Model diagnostics

![Model MAE comparison](../results/mcn_model_mae_comparison.png)

This is the leaderboard corresponding to the numerical table above.

![Banzhaf predicted versus exact](../results/mcn_banzhaf_prediction_scatter.png)

Each point is one agent in one test game; the diagonal represents a perfect
prediction.

![Shapley predicted versus exact](../results/mcn_shapley_prediction_scatter.png)

The same diagnostic is shown for Shapley-Shubik labels.

Additional per-agent errors, training curves, and single-game comparisons are
stored in `results/`.

## Other MCN runs

Train both targets or select one:

```bash
python train_models.py --data data/mcn_games.npz --game-type mcn
python train_models.py --data data/mcn_games.npz --game-type mcn --indices banzhaf
python train_models.py --data data/mcn_games.npz --game-type mcn --indices shapley
```

Predict a saved dataset example:

```bash
python predict.py \
  --game-type mcn \
  --data data/mcn_games.npz \
  --example-index 0
```

Predict a fresh game with the trained shape:

```bash
python predict.py \
  --game-type mcn \
  --num-agents 6 \
  --num-rules 12 \
  --rule-generator uniform \
  --value-generator uniform \
  --seed 123
```

## Weighted-voting workflow

The earlier weighted-voting experiment remains available:

```bash
python generate_data.py \
  --game-type weighted \
  --num-games 20000 \
  --num-agents 8 \
  --seed 42 \
  --output data/voting_games.csv

python train_models.py \
  --data data/voting_games.csv \
  --game-type weighted \
  --epochs 300 \
  --sklearn-max-iter 300 \
  --n-estimators 40

python predict.py \
  --game-type weighted \
  --weights 4 2 7 1 5 3 6 2 \
  --quota 16
```

The Monte Carlo demo estimates weighted-game Banzhaf power with confidence
intervals:

```bash
python monte_carlo_demo.py \
  --weights 4 2 7 1 5 3 6 2 \
  --quota 16 \
  --samples 10000 \
  --seed 42
```

## Generated artifacts

Datasets, model binaries, CSV metrics, and histories are ignored by Git because
they can be regenerated. PNG figures in `results/` are versioned for the
documentation.

Important outputs include:

```text
data/mcn_games.npz
results/mcn_model_metrics.csv
results/mcn_banzhaf_numpy_mlp_history.csv
results/mcn_shapley_numpy_mlp_history.csv
models/mcn_<index>_numpy_mlp.npz
models/mcn_<index>_scratch_<model>.pkl
models/mcn_<index>_sklearn_<model>.pkl
```

## Reproducibility and limits

- Seeds control generation, sampling, splitting, NumPy initialization, and the
  scikit-learn estimators.
- Exact MCN calculation grows exponentially with the number of agents.
- Monte Carlo labels add sampling error in exchange for scalability.
- Output labels are normalized influence distributions by default.
- The default unsigned marginal change counts both fulfilled and broken rules.
- A model trained on one MCN tensor shape cannot predict another shape.
- The NumPy MLP and scratch forests prioritize readability over speed.
- Low-level numerical libraries and parallel scheduling can still cause tiny
  cross-platform differences despite pinned top-level packages.

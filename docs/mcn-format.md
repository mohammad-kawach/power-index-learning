# MCN Format

InfluenceNet Mini stores Marginal Contribution Networks as numeric rule tensors.
This page specifies the representation, generators, and saved dataset format.

## One game

One game is a matrix with shape:

```text
(num_rules, 2 * num_agents + 1)
```

The columns are:

| Part | Width | Meaning |
| --- | ---: | --- |
| `req_*` | `num_agents` | Binary flags for agents that must be present |
| `ban_*` | `num_agents` | Binary flags for agents that must be absent |
| `value` | 1 | Value added when the rule is satisfied |

An agent cannot be both required and banned by the same rule.

For three agents, the rule set from the paper is encoded as:

| Rule | req_0 | req_1 | req_2 | ban_0 | ban_1 | ban_2 | value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `{a and b}` | 1 | 1 | 0 | 0 | 0 | 0 | 3 |
| `{a and not c}` | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| `{b and not c}` | 0 | 1 | 0 | 0 | 0 | 1 | 2 |

Coalition values follow from summing the active rules:

| Coalition | Active rules | Value |
| --- | --- | ---: |
| `{a}` | `{a and not c}` | 1 |
| `{b}` | `{b and not c}` | 2 |
| `{a, b}` | all three rules | 6 |
| `{a, c}` | none | 0 |

## A dataset

A batch of games is a real 3D tensor:

```text
rules.shape == (num_games, num_rules, 2 * num_agents + 1)
```

For 500 games with 6 agents and 12 rules, the shape is `(500, 12, 13)`.
The tensor is flattened only when passed to the current 2D regressors.

The compressed `.npz` archive contains:

| Key | Contents |
| --- | --- |
| `rules` | 3D rule tensor |
| `banzhaf_targets` | normalized per-agent Banzhaf labels |
| `shapley_targets` | normalized per-agent Shapley-Shubik labels |
| `num_games`, `num_rules`, `num_agents` | dimensions |
| `seed` | dataset RNG seed |
| `rule_generator`, `value_generator`, `p` | generator settings |
| `label_method` | `exact` or `monte_carlo` |
| `monte_carlo_samples`, `monte_carlo_batch_size` | sampling settings |

## Rule generators

The three generator families are adapted from the InfluenceNet paper:

| Option | Behaviour in this implementation |
| --- | --- |
| `uniform` | Samples two uniform arrays, thresholds required flags, then assigns non-required agents to banned flags through the second array |
| `coin_flip` | Repeatedly selects an agent and assigns it to the required or banned role |
| `gaussian_mixture` | Draws per-rule Gaussian means and standard deviations from gamma distributions before thresholding samples |

Select one with `--rule-generator`.

## Rule-value generators

| Option | Behaviour in this implementation |
| --- | --- |
| `uniform` | Sets every rule value to `1` |
| `low_variance` | Samples a clipped normal distribution with mean `1.0` and standard deviation `0.15` |
| `high_variance` | Samples a clipped normal distribution with mean `1.0` and standard deviation `0.75` |

The clipping floor is `0.05`. These numerical choices are repository-specific
implementations of the paper's uniform, low-variance, and high-variance value
families; they should not be read as reported constants from the publication.

## Exact and Monte Carlo labels

Use exact enumeration for small games:

```bash
python generate_data.py \
  --game-type mcn \
  --num-games 500 \
  --num-agents 6 \
  --num-rules 12 \
  --label-method exact \
  --seed 42 \
  --output data/mcn_games.npz
```

Use sampling for larger games:

```bash
python generate_data.py \
  --game-type mcn \
  --num-games 2000 \
  --num-agents 12 \
  --num-rules 20 \
  --rule-generator coin_flip \
  --value-generator high_variance \
  --label-method monte_carlo \
  --monte-carlo-samples 10000 \
  --seed 42 \
  --output data/mcn_games.npz
```

Exact runtime grows exponentially with the number of agents. For a quick
sampled trial, reduce both `--num-games` and `--monte-carlo-samples`.

## Shape compatibility

The model input width is `num_rules * (2 * num_agents + 1)`. A saved model can
therefore predict only MCNs with the same agent and rule counts used during its
training. The prediction command checks this and reports a feature-count error
for incompatible shapes.

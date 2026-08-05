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

## Model feature sets

Training supports two MCN feature representations:

| Option | Contents |
| --- | --- |
| `raw` | Only the flattened rule tensor, with width `num_rules * (2 * num_agents + 1)` |
| `augmented` | The raw tensor plus deterministic MCN aggregate features |

The aggregate block contains seven per-agent summaries and sixteen global rule
statistics:

```text
aggregate_width = 7 * num_agents + 16
augmented_width = raw_width + aggregate_width
```

For the canonical 6-agent, 12-rule shape, `raw_width` is `156` and
`augmented_width` is `214`. The per-agent summaries cover required, banned,
and mentioned frequencies, role-specific value shares, total role value share,
and signed role value share. The global statistics summarize rule values and
required/banned/mentioned rule complexity.

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

The model input width depends on both the tensor shape and the feature set. A
saved model can therefore predict only MCNs with the same agent count, rule
count, and compatible feature representation used during training. The
prediction command defaults to `--mcn-feature-set auto`, which selects `raw` or
`augmented` by matching the saved model width when possible.

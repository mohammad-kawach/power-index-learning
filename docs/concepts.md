# Concepts

This page introduces the cooperative-game concepts used by InfluenceNet Mini.
For the concrete tensor layout, see [MCN format](mcn-format.md).

## Agents and coalitions

An **agent** is one participant in a cooperative game: for example, a political
party, a board member, or a network node. A **coalition** is a subset of the
agents working together.

With agents `a`, `b`, and `c`, the eight coalitions are:

```text
{}
{a}
{b}
{c}
{a, b}
{a, c}
{b, c}
{a, b, c}
```

An `n`-agent game has `2^n` coalitions. This exponential growth is why exact
power-index calculation becomes expensive as the agent count increases.

## Weighted voting games

The older workflow represents a game as:

```text
[quota; weight_0, weight_1, ..., weight_n]
```

A coalition wins when its total weight reaches the quota. This representation
is intuitive but cannot express every local interaction between agents.

## Marginal Contribution Networks

A Marginal Contribution Network (MCN) assigns values through rules:

```text
if these agents are present
and these other agents are absent
then add this rule value to the coalition
```

The value of a coalition is the sum of every rule it satisfies. Required and
banned agents let MCNs represent positive interactions, exclusions, and local
patterns that a single quota and weight vector cannot capture.

For example:

```text
{a and b}      -> 3
{a and not c}  -> 1
{b and not c}  -> 2
```

This gives `v({a}) = 1`, `v({b}) = 2`, and `v({a, b}) = 6`.

## Power indices

Power indices quantify how strongly each agent changes coalition outcomes.

### Banzhaf

The Banzhaf measure averages an agent's marginal contribution over coalitions
of the other agents, treating those coalitions as equally likely.

For agent `i` in a game with agent set `N`, the signed raw value is:

```text
beta_i = (1 / 2^(|N|-1)) * sum over S subset N\{i} of [v(S union {i}) - v(S)]
```

### Shapley-Shubik

The Shapley-Shubik measure averages marginal contribution over all orders in
which agents could join. Equivalently, it weights each predecessor coalition by
the number of permutations that produce it:

```text
phi_i = sum over S subset N\{i}
        |S|! (|N|-|S|-1)! / |N|! * [v(S union {i}) - v(S)]
```

The distinction is the probability model: Banzhaf weights coalitions equally,
whereas Shapley-Shubik weights joining orders equally.

## The default label convention

MCN rules can be activated or broken when an agent joins. InfluenceNet Mini
uses the absolute marginal change by default, so both events count as
influence. It then normalizes each game's agent vector to sum to one when the
raw total is positive. This paper-inspired, unsigned convention is useful for
the learning task but is not identical to the signed value used in every
cooperative-game treatment.

The Python API makes the choice explicit:

```python
exact_power_indices(rules, absolute=True, normalize=True)   # project default
exact_power_indices(rules, absolute=False, normalize=False) # signed raw values
```

Results should therefore state the `absolute` and `normalize` settings when
they are compared with another implementation or publication.

## Exact and sampled calculation

Exact calculation enumerates all predecessor coalitions and is appropriate for
small games. Monte Carlo calculation samples coalitions and permutations, which
trades deterministic exactness for scalability. The project supports both and
uses a seed to make its sampled runs repeatable.

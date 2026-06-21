import numpy as np
from itertools import product

NUM_AGENTS = 5


def coalition_value(coalition: np.ndarray, weights: np.ndarray, quota: int) -> int:
    """Return 1 if a coalition is winning, otherwise 0."""
    return int(np.dot(coalition, weights) >= quota)


def exact_banzhaf(weights: np.ndarray, quota: int, num_agents: int = NUM_AGENTS) -> np.ndarray:
    """
    Exact normalized Banzhaf power index for small weighted voting games.

    For every agent, we check all coalitions without that agent. If adding the
    agent changes the coalition from losing to winning, that agent is critical.
    """
    power = np.zeros(num_agents, dtype=float)

    for agent in range(num_agents):
        other_agents = [i for i in range(num_agents) if i != agent]

        for bits in product([0, 1], repeat=num_agents - 1):
            coalition_without = np.zeros(num_agents, dtype=float)

            for idx, other_agent in enumerate(other_agents):
                coalition_without[other_agent] = bits[idx]

            coalition_with = coalition_without.copy()
            coalition_with[agent] = 1

            before = coalition_value(coalition_without, weights, quota)
            after = coalition_value(coalition_with, weights, quota)

            if after - before == 1:
                power[agent] += 1

    total = power.sum()
    if total > 0:
        power = power / total

    return power


def monte_carlo_banzhaf(
    weights: np.ndarray,
    quota: float,
    num_samples: int = 10_000,
    seed: int | None = None,
    batch_size: int = 10_000,
) -> np.ndarray:
    """Approximate the normalized Banzhaf index by sampling coalitions.

    Each sampled coalition includes every agent independently with probability
    1/2. For agent ``i``, its sampled membership is ignored, giving a uniformly
    random coalition of the other agents. The agent is critical when adding it
    changes that coalition from losing to winning. The estimated critical
    probabilities are normalized to sum to one, matching :func:`exact_banzhaf`.

    Sampling is processed in batches so memory use is bounded for games with
    many agents. Passing ``seed`` makes the estimate reproducible.
    """
    weights = np.asarray(weights, dtype=float)
    try:
        quota = float(quota)
    except (TypeError, ValueError) as error:
        raise ValueError("quota must be a finite number") from error

    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if not np.isfinite(quota):
        raise ValueError("quota must be a finite number")
    if not isinstance(num_samples, (int, np.integer)) or isinstance(
        num_samples, bool
    ):
        raise ValueError("num_samples must be a positive integer")
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not isinstance(batch_size, (int, np.integer)) or isinstance(
        batch_size, bool
    ):
        raise ValueError("batch_size must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    rng = np.random.default_rng(seed)
    critical_counts = np.zeros(weights.size, dtype=np.int64)

    samples_remaining = num_samples
    while samples_remaining:
        current_batch_size = min(batch_size, samples_remaining)
        coalitions = rng.integers(
            0,
            2,
            size=(current_batch_size, weights.size),
            dtype=np.int8,
        )
        coalition_values = coalitions @ weights

        # Removing i when present, or leaving the coalition unchanged when it
        # is absent, produces a uniform sample over coalitions not containing i.
        values_without_agent = coalition_values[:, None] - coalitions * weights
        is_critical = (values_without_agent < quota) & (
            values_without_agent + weights >= quota
        )
        critical_counts += np.count_nonzero(is_critical, axis=0)
        samples_remaining -= current_batch_size

    estimated_power = critical_counts.astype(float) / num_samples
    total_power = estimated_power.sum()
    if total_power > 0:
        estimated_power /= total_power

    return estimated_power

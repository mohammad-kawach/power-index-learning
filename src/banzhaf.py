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

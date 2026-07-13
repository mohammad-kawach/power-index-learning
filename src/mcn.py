"""Rule-based Marginal Contribution Network utilities.

The paper represents each game as a tensor with shape
``(num_rules, 2 * num_agents + 1)``. For every rule, the first ``num_agents``
columns are required agents, the next ``num_agents`` columns are banned agents,
and the final column is the rule value.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial

import numpy as np


@dataclass(frozen=True)
class MCNPowerResult:
    """Banzhaf and Shapley-style influence scores for one MCN game."""

    banzhaf: np.ndarray
    shapley: np.ndarray
    raw_banzhaf: np.ndarray
    raw_shapley: np.ndarray


def validate_rule_tensor(rules: np.ndarray) -> np.ndarray:
    """Return a validated MCN rule tensor as ``float`` values."""
    rules = np.asarray(rules, dtype=float)
    if rules.ndim != 2:
        raise ValueError("rules must have shape (num_rules, 2 * num_agents + 1)")
    if rules.shape[0] == 0:
        raise ValueError("rules must contain at least one rule")
    if rules.shape[1] < 5 or rules.shape[1] % 2 != 1:
        raise ValueError("rule width must be 2 * num_agents + 1 with at least two agents")
    if not np.all(np.isfinite(rules)):
        raise ValueError("rules must contain only finite values")

    num_agents = infer_num_agents_from_rules(rules)
    requirements = rules[:, :num_agents]
    bans = rules[:, num_agents : 2 * num_agents]
    if not np.all(np.isin(requirements, (0, 1))) or not np.all(np.isin(bans, (0, 1))):
        raise ValueError("required and banned columns must be binary 0/1 indicators")
    if np.any((requirements == 1) & (bans == 1)):
        raise ValueError("the same agent cannot be both required and banned in one rule")
    return rules


def validate_rule_batch(rule_tensor: np.ndarray) -> np.ndarray:
    """Return a validated batch of MCN rule tensors."""
    rule_tensor = np.asarray(rule_tensor, dtype=float)
    if rule_tensor.ndim != 3:
        raise ValueError("rule_tensor must have shape (games, rules, 2 * agents + 1)")
    if rule_tensor.shape[0] == 0:
        raise ValueError("rule_tensor must contain at least one game")
    if rule_tensor.shape[1] == 0:
        raise ValueError("rule_tensor must contain at least one rule per game")
    if rule_tensor.shape[2] < 5 or rule_tensor.shape[2] % 2 != 1:
        raise ValueError("rule width must be 2 * num_agents + 1 with at least two agents")
    if not np.all(np.isfinite(rule_tensor)):
        raise ValueError("rule_tensor must contain only finite values")
    num_agents = (rule_tensor.shape[2] - 1) // 2
    requirements = rule_tensor[:, :, :num_agents]
    bans = rule_tensor[:, :, num_agents : 2 * num_agents]
    if not np.all(np.isin(requirements, (0, 1))) or not np.all(np.isin(bans, (0, 1))):
        raise ValueError("required and banned columns must be binary 0/1 indicators")
    if np.any((requirements == 1) & (bans == 1)):
        raise ValueError("the same agent cannot be both required and banned in one rule")
    return rule_tensor


def infer_num_agents_from_rules(rules: np.ndarray) -> int:
    """Infer the number of agents from one rule matrix."""
    return (np.asarray(rules).shape[1] - 1) // 2


def split_rules(rules: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a rule matrix into required indicators, banned indicators, and values."""
    rules = validate_rule_tensor(rules)
    num_agents = infer_num_agents_from_rules(rules)
    return (
        rules[:, :num_agents].astype(bool),
        rules[:, num_agents : 2 * num_agents].astype(bool),
        rules[:, -1].astype(float),
    )


def _coalition_value_parts(
    coalition: np.ndarray,
    required: np.ndarray,
    banned: np.ndarray,
    values: np.ndarray,
) -> float:
    required_met = np.all(~required | coalition, axis=1)
    banned_absent = np.all(~banned | ~coalition, axis=1)
    return float(values[required_met & banned_absent].sum())


def _marginal_contribution_parts(
    coalition: np.ndarray,
    agent: int,
    required: np.ndarray,
    banned: np.ndarray,
    values: np.ndarray,
    *,
    absolute: bool,
) -> float:
    coalition = coalition.copy()
    coalition[agent] = False
    without = _coalition_value_parts(coalition, required, banned, values)
    coalition[agent] = True
    delta = _coalition_value_parts(coalition, required, banned, values) - without
    return abs(delta) if absolute else float(delta)


def coalition_value(coalition: np.ndarray, rules: np.ndarray) -> float:
    """Return the total value of all rules satisfied by ``coalition``."""
    required, banned, values = split_rules(rules)
    coalition = np.asarray(coalition, dtype=bool)
    if coalition.shape != (required.shape[1],):
        raise ValueError("coalition length must match the number of agents")
    return _coalition_value_parts(coalition, required, banned, values)


def rule_status(coalition: np.ndarray, rules: np.ndarray) -> np.ndarray:
    """Return a boolean vector showing which rules are active."""
    required, banned, _ = split_rules(rules)
    coalition = np.asarray(coalition, dtype=bool)
    if coalition.shape != (required.shape[1],):
        raise ValueError("coalition length must match the number of agents")
    return np.all(~required | coalition, axis=1) & np.all(~banned | ~coalition, axis=1)


def marginal_contribution(
    rules: np.ndarray,
    coalition: np.ndarray,
    agent: int,
    *,
    absolute: bool = True,
) -> float:
    """Return the value change caused by adding ``agent`` to ``coalition``.

    ``absolute=True`` follows the paper's "fulfilled or broken rule" counting:
    an agent receives influence when adding it either activates or disables a
    weighted rule. Set ``absolute=False`` for the signed cooperative-game value.
    """
    rules = validate_rule_tensor(rules)
    num_agents = infer_num_agents_from_rules(rules)
    if agent < 0 or agent >= num_agents:
        raise ValueError("agent index out of range")
    coalition = np.asarray(coalition, dtype=bool).copy()
    if coalition.shape != (num_agents,):
        raise ValueError("coalition length must match the number of agents")
    required, banned, values = split_rules(rules)
    return _marginal_contribution_parts(
        coalition, agent, required, banned, values, absolute=absolute
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total > 0:
        return values / total
    return np.zeros_like(values)


def exact_power_indices(
    rules: np.ndarray,
    *,
    absolute: bool = True,
    normalize: bool = True,
) -> MCNPowerResult:
    """Compute exact Banzhaf and Shapley-style MCN influence scores.

    Exact enumeration is useful for small educational datasets. Runtime is
    exponential in the number of agents, so use ``monte_carlo_power_indices``
    for larger games.
    """
    rules = validate_rule_tensor(rules)
    num_agents = infer_num_agents_from_rules(rules)
    required, banned, values = split_rules(rules)
    raw_banzhaf = np.zeros(num_agents, dtype=float)
    raw_shapley = np.zeros(num_agents, dtype=float)
    coefficients = np.array(
        [
            factorial(size) * factorial(num_agents - size - 1) / factorial(num_agents)
            for size in range(num_agents)
        ]
    )

    for agent in range(num_agents):
        others = np.delete(np.arange(num_agents), agent)
        for bits in product((False, True), repeat=num_agents - 1):
            coalition = np.zeros(num_agents, dtype=bool)
            coalition[others] = bits
            delta = _marginal_contribution_parts(
                coalition, agent, required, banned, values, absolute=absolute
            )
            raw_banzhaf[agent] += delta
            raw_shapley[agent] += coefficients[int(np.sum(bits))] * delta
        raw_banzhaf[agent] /= 2 ** (num_agents - 1)

    if normalize:
        return MCNPowerResult(
            banzhaf=_normalize(raw_banzhaf),
            shapley=_normalize(raw_shapley),
            raw_banzhaf=raw_banzhaf,
            raw_shapley=raw_shapley,
        )
    return MCNPowerResult(
        banzhaf=raw_banzhaf,
        shapley=raw_shapley,
        raw_banzhaf=raw_banzhaf,
        raw_shapley=raw_shapley,
    )


def monte_carlo_power_indices(
    rules: np.ndarray,
    num_samples: int = 10_000,
    seed: int | None = None,
    *,
    absolute: bool = True,
    normalize: bool = True,
    batch_size: int = 2_048,
) -> MCNPowerResult:
    """Approximate MCN Banzhaf and Shapley-style scores by sampling."""
    if (
        not isinstance(num_samples, (int, np.integer))
        or isinstance(num_samples, bool)
        or num_samples <= 0
    ):
        raise ValueError("num_samples must be a positive integer")
    if (
        not isinstance(batch_size, (int, np.integer))
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    rules = validate_rule_tensor(rules)
    num_agents = infer_num_agents_from_rules(rules)
    required, banned, values = split_rules(rules)
    rng = np.random.default_rng(seed)
    raw_banzhaf = np.zeros(num_agents, dtype=float)
    raw_shapley = np.zeros(num_agents, dtype=float)
    role = required | banned
    signed_values = values[:, None] * (required.astype(float) - banned.astype(float))

    samples_done = 0
    while samples_done < num_samples:
        current_batch = min(batch_size, num_samples - samples_done)
        coalitions = rng.integers(
            0, 2, size=(current_batch, num_agents), dtype=np.int8
        ).astype(bool)

        req_missing = ((~coalitions)[:, None, :] & required[None, :, :]).sum(axis=2)
        ban_present = (coalitions[:, None, :] & banned[None, :, :]).sum(axis=2)
        req_missing_other = req_missing[:, :, None] - (
            (~coalitions)[:, None, :] & required[None, :, :]
        )
        ban_present_other = ban_present[:, :, None] - (
            coalitions[:, None, :] & banned[None, :, :]
        )
        changed = (
            (req_missing_other == 0)
            & (ban_present_other == 0)
            & role[None, :, :]
        )
        banzhaf_delta = np.sum(changed * signed_values[None, :, :], axis=1)
        raw_banzhaf += (
            np.abs(banzhaf_delta).sum(axis=0) if absolute else banzhaf_delta.sum(axis=0)
        )

        random_order_keys = rng.random((current_batch, num_agents))
        positions = np.argsort(np.argsort(random_order_keys, axis=1), axis=1)
        agent_positions = positions[:, None, :]

        required_positions = np.where(required[None, :, :], agent_positions, -1)
        banned_positions = np.where(banned[None, :, :], agent_positions, num_agents)
        required_partition = np.partition(required_positions, -2, axis=2)
        banned_partition = np.partition(banned_positions, 1, axis=2)
        latest_required = required_partition[:, :, -1]
        second_latest_required = required_partition[:, :, -2]
        earliest_banned = banned_partition[:, :, 0]
        second_earliest_banned = banned_partition[:, :, 1]

        latest_other_required = np.where(
            required[None, :, :] & (agent_positions == latest_required[:, :, None]),
            second_latest_required[:, :, None],
            latest_required[:, :, None],
        )
        earliest_other_banned = np.where(
            banned[None, :, :] & (agent_positions == earliest_banned[:, :, None]),
            second_earliest_banned[:, :, None],
            earliest_banned[:, :, None],
        )
        changed_at_join = (
            (latest_other_required < agent_positions)
            & (earliest_other_banned > agent_positions)
            & role[None, :, :]
        )
        batch_shapley = np.sum(changed_at_join * signed_values[None, :, :], axis=1)

        raw_shapley += (
            np.abs(batch_shapley).sum(axis=0)
            if absolute
            else batch_shapley.sum(axis=0)
        )
        samples_done += current_batch

    raw_banzhaf /= num_samples
    raw_shapley /= num_samples
    if normalize:
        return MCNPowerResult(
            banzhaf=_normalize(raw_banzhaf),
            shapley=_normalize(raw_shapley),
            raw_banzhaf=raw_banzhaf,
            raw_shapley=raw_shapley,
        )
    return MCNPowerResult(
        banzhaf=raw_banzhaf,
        shapley=raw_shapley,
        raw_banzhaf=raw_banzhaf,
        raw_shapley=raw_shapley,
    )


def generate_rule_values(
    rng: np.random.Generator,
    num_rules: int,
    value_generator: str = "uniform",
) -> np.ndarray:
    """Generate positive rule weights using the paper's three value families."""
    if value_generator == "uniform":
        return np.ones(num_rules, dtype=float)
    if value_generator == "low_variance":
        return np.clip(rng.normal(loc=1.0, scale=0.15, size=num_rules), 0.05, None)
    if value_generator == "high_variance":
        return np.clip(rng.normal(loc=1.0, scale=0.75, size=num_rules), 0.05, None)
    raise ValueError(
        "value_generator must be 'uniform', 'low_variance', or 'high_variance'"
    )


def generate_random_rules(
    num_rules: int,
    num_agents: int,
    *,
    rng: np.random.Generator,
    rule_generator: str = "uniform",
    value_generator: str = "uniform",
    p: float = 0.5,
    num_coins: int | None = None,
    gamma_shape: float = 2.0,
    gamma_rate: float = 2.0,
) -> np.ndarray:
    """Generate one random MCN rule matrix.

    The supported rule generators correspond to the paper's uniform sampling,
    coin-flip assignment, and probabilistic mixture-of-Gaussians strategies.
    """
    if num_rules <= 0:
        raise ValueError("num_rules must be positive")
    if num_agents < 2:
        raise ValueError("num_agents must be at least 2")
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if gamma_shape <= 0 or gamma_rate <= 0:
        raise ValueError("gamma_shape and gamma_rate must be positive")

    required = np.zeros((num_rules, num_agents), dtype=int)
    banned = np.zeros((num_rules, num_agents), dtype=int)

    if rule_generator == "uniform":
        x = rng.random((num_rules, num_agents))
        y = rng.random((num_rules, num_agents))
        required = (x >= p).astype(int)
        banned = ((required == 0) & (y >= p)).astype(int)
    elif rule_generator == "coin_flip":
        coins = num_coins if num_coins is not None else max(1, num_agents // 2)
        if coins <= 0:
            raise ValueError("num_coins must be positive")
        for rule_index in range(num_rules):
            chosen_agents = rng.integers(0, num_agents, size=coins)
            assignments = rng.integers(0, 2, size=coins)
            for agent, assignment in zip(chosen_agents, assignments):
                if assignment:
                    required[rule_index, agent] = 1
                    banned[rule_index, agent] = 0
                else:
                    banned[rule_index, agent] = 1
                    required[rule_index, agent] = 0
    elif rule_generator == "gaussian_mixture":
        scale = 1.0 / gamma_rate
        means = rng.gamma(shape=gamma_shape, scale=scale, size=num_rules)
        stds = rng.gamma(shape=gamma_shape, scale=scale, size=num_rules)
        x = rng.normal(means[:, None], stds[:, None], size=(num_rules, num_agents))
        y = rng.normal(means[:, None], stds[:, None], size=(num_rules, num_agents))
        required = (x >= p).astype(int)
        banned = ((required == 0) & (y >= p)).astype(int)
    else:
        raise ValueError(
            "rule_generator must be 'uniform', 'coin_flip', or 'gaussian_mixture'"
        )

    values = generate_rule_values(rng, num_rules, value_generator)
    return np.concatenate([required, banned, values[:, None]], axis=1).astype(float)


def generate_dataset(
    num_games: int = 2_000,
    num_rules: int = 20,
    num_agents: int = 8,
    *,
    seed: int = 42,
    rule_generator: str = "uniform",
    value_generator: str = "uniform",
    p: float = 0.5,
    label_method: str = "exact",
    monte_carlo_samples: int = 10_000,
    monte_carlo_batch_size: int = 2_048,
    num_coins: int | None = None,
    gamma_shape: float = 2.0,
    gamma_rate: float = 2.0,
) -> dict[str, np.ndarray | str | int | float]:
    """Generate a labeled batch of MCN games."""
    if num_games <= 0:
        raise ValueError("num_games must be positive")
    if label_method not in {"exact", "monte_carlo"}:
        raise ValueError("label_method must be 'exact' or 'monte_carlo'")

    rng = np.random.default_rng(seed)
    games = np.zeros((num_games, num_rules, 2 * num_agents + 1), dtype=float)
    banzhaf_targets = np.zeros((num_games, num_agents), dtype=float)
    shapley_targets = np.zeros((num_games, num_agents), dtype=float)

    for game_index in range(num_games):
        rules = generate_random_rules(
            num_rules,
            num_agents,
            rng=rng,
            rule_generator=rule_generator,
            value_generator=value_generator,
            p=p,
            num_coins=num_coins,
            gamma_shape=gamma_shape,
            gamma_rate=gamma_rate,
        )
        if label_method == "exact":
            result = exact_power_indices(rules)
        else:
            result = monte_carlo_power_indices(
                rules,
                num_samples=monte_carlo_samples,
                seed=int(rng.integers(0, 1_000_000_000)),
                batch_size=monte_carlo_batch_size,
            )
        games[game_index] = rules
        banzhaf_targets[game_index] = result.banzhaf
        shapley_targets[game_index] = result.shapley
        if num_games >= 10 and (game_index + 1) % max(1, num_games // 10) == 0:
            print(f"  {game_index + 1}/{num_games}")

    return {
        "rules": games,
        "banzhaf_targets": banzhaf_targets,
        "shapley_targets": shapley_targets,
        "game_type": "mcn",
        "num_games": num_games,
        "num_rules": num_rules,
        "num_agents": num_agents,
        "seed": seed,
        "rule_generator": rule_generator,
        "value_generator": value_generator,
        "p": p,
        "label_method": label_method,
        "monte_carlo_samples": monte_carlo_samples,
        "monte_carlo_batch_size": monte_carlo_batch_size,
    }

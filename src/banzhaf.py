"""Exact and Monte Carlo power-index calculations for weighted voting games."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb, factorial
from statistics import NormalDist

import numpy as np


# Kept as the default for old imports. All calculation functions infer the
# actual number of agents from ``weights``.
NUM_AGENTS = 5


@dataclass(frozen=True)
class MonteCarloResult:
    """A normalized Monte Carlo estimate with uncertainty information."""

    estimate: np.ndarray
    standard_error: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    raw_estimate: np.ndarray
    raw_standard_error: np.ndarray
    num_samples: int
    confidence_level: float
    method: str


def _validate_game(weights, quota):
    weights = np.asarray(weights, dtype=float)
    try:
        quota = float(quota)
    except (TypeError, ValueError) as error:
        raise ValueError("quota must be a finite number") from error
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must contain only finite, non-negative values")
    if not np.isfinite(quota):
        raise ValueError("quota must be a finite number")
    return weights, quota


def coalition_value(coalition: np.ndarray, weights: np.ndarray, quota: float) -> int:
    """Return 1 if a coalition is winning, otherwise 0."""
    return int(np.dot(coalition, weights) >= quota)


def exact_power_indices(weights: np.ndarray, quota: float) -> tuple[np.ndarray, np.ndarray]:
    """Return exact normalized Banzhaf and Shapley--Shubik indices.

    This enumerates each coalition once per agent. It is intended for label
    generation in small games; complexity is exponential in the agent count.
    """
    weights, quota = _validate_game(weights, quota)
    n = weights.size
    swings = np.zeros(n, dtype=float)
    shapley = np.zeros(n, dtype=float)
    shapley_coefficients = np.array(
        [factorial(size) * factorial(n - size - 1) / factorial(n) for size in range(n)]
    )

    for agent in range(n):
        others = np.delete(np.arange(n), agent)
        for bits in product((0, 1), repeat=n - 1):
            coalition_weight = float(np.dot(bits, weights[others]))
            if coalition_weight < quota <= coalition_weight + weights[agent]:
                size = int(sum(bits))
                swings[agent] += 1.0
                shapley[agent] += shapley_coefficients[size]

    banzhaf = swings / swings.sum() if swings.sum() else swings
    # For ordinary feasible weighted voting games this already sums to one.
    # Normalization also gives stable behavior for unusual quota edge cases.
    if shapley.sum() > 0:
        shapley /= shapley.sum()
    return banzhaf, shapley


def exact_banzhaf(
    weights: np.ndarray, quota: float, num_agents: int | None = None
) -> np.ndarray:
    """Return the exact normalized Banzhaf power index."""
    weights = np.asarray(weights)
    if num_agents is not None and num_agents != weights.size:
        raise ValueError("num_agents must match the number of weights")
    return exact_power_indices(weights, quota)[0]


def exact_shapley_shubik(weights: np.ndarray, quota: float) -> np.ndarray:
    """Return the exact Shapley--Shubik index for a weighted voting game."""
    return exact_power_indices(weights, quota)[1]


def _validate_sampling(num_samples, batch_size, confidence_level, method):
    for name, value in (("num_samples", num_samples), ("batch_size", batch_size)):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if method not in {"plain", "antithetic", "stratified"}:
        raise ValueError("method must be 'plain', 'antithetic', or 'stratified'")


def _normalize_with_uncertainty(raw, covariance, confidence_level):
    total = raw.sum()
    if total <= 0:
        zeros = np.zeros_like(raw)
        return zeros, zeros, zeros, zeros

    estimate = raw / total
    # Delta method for g_i(x) = x_i / sum(x).
    jacobian = (np.eye(raw.size) * total - raw[:, None]) / total**2
    normalized_covariance = jacobian @ covariance @ jacobian.T
    standard_error = np.sqrt(np.maximum(np.diag(normalized_covariance), 0.0))
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    ci_low = np.clip(estimate - z_score * standard_error, 0.0, 1.0)
    ci_high = np.clip(estimate + z_score * standard_error, 0.0, 1.0)
    return estimate, standard_error, ci_low, ci_high


def _plain_or_antithetic_samples(weights, quota, num_samples, batch_size, rng, method):
    n = weights.size
    sum_observations = np.zeros(n)
    cross_products = np.zeros((n, n))
    units = 0
    samples_remaining = num_samples

    while samples_remaining:
        requested = min(batch_size, samples_remaining)
        if method == "plain":
            coalitions = rng.integers(0, 2, size=(requested, n), dtype=np.int8)
            coalition_weights = coalitions @ weights
            without = coalition_weights[:, None] - coalitions * weights
            observations = ((without < quota) & (without + weights >= quota)).astype(float)
        else:
            # Each statistical observation is the average of a coalition and
            # its complement. This costs two game evaluations but preserves
            # the target expectation and often lowers its variance.
            coalitions = rng.integers(0, 2, size=(requested, n), dtype=np.int8)
            pair_observations = []
            for batch in (coalitions, 1 - coalitions):
                coalition_weights = batch @ weights
                without = coalition_weights[:, None] - batch * weights
                pair_observations.append(
                    ((without < quota) & (without + weights >= quota)).astype(float)
                )
            observations = (pair_observations[0] + pair_observations[1]) / 2

        sum_observations += observations.sum(axis=0)
        cross_products += observations.T @ observations
        units += len(observations)
        samples_remaining -= requested

    raw = sum_observations / units
    if units > 1:
        sample_cov = (cross_products - units * np.outer(raw, raw)) / (units - 1)
        covariance = sample_cov / units
    else:
        covariance = np.zeros((n, n))
    return raw, covariance


def _stratified_samples(weights, quota, num_samples, rng):
    """Estimate each swing probability by stratifying coalition cardinality."""
    n = weights.size
    raw = np.zeros(n)
    variances = np.zeros(n)
    probabilities = np.array([comb(n - 1, k) / 2 ** (n - 1) for k in range(n)])
    if num_samples < n:
        raise ValueError("stratified sampling requires num_samples >= number of agents")
    allocation = np.ones(n, dtype=int)
    remaining = num_samples - n
    allocation += np.floor(remaining * probabilities).astype(int)
    remainder = num_samples - allocation.sum()
    if remainder:
        allocation[np.argsort(probabilities)[-remainder:]] += 1

    for agent in range(n):
        others = np.delete(np.arange(n), agent)
        for size, draws in enumerate(allocation):
            if draws == 0:
                continue
            indicators = np.empty(draws)
            for draw in range(draws):
                chosen = rng.choice(others, size=size, replace=False)
                coalition_weight = weights[chosen].sum()
                indicators[draw] = coalition_weight < quota <= coalition_weight + weights[agent]
            stratum_mean = indicators.mean()
            raw[agent] += probabilities[size] * stratum_mean
            if draws > 1:
                variances[agent] += probabilities[size] ** 2 * indicators.var(ddof=1) / draws
    return raw, np.diag(variances)


def monte_carlo_banzhaf(
    weights: np.ndarray,
    quota: float,
    num_samples: int = 10_000,
    seed: int | None = None,
    batch_size: int = 10_000,
    *,
    confidence_level: float = 0.95,
    method: str = "plain",
    return_result: bool = False,
) -> np.ndarray | MonteCarloResult:
    """Estimate normalized Banzhaf power using sampled coalitions.

    ``method`` may be ``plain`` (iid coalitions), ``antithetic`` (a coalition
    is paired with its complement), or ``stratified`` (samples are allocated
    across coalition sizes). The default return value remains the normalized
    vector for backward compatibility. Set ``return_result=True`` to receive
    standard errors and normal-approximation confidence intervals.
    """
    weights, quota = _validate_game(weights, quota)
    _validate_sampling(num_samples, batch_size, confidence_level, method)
    rng = np.random.default_rng(seed)

    if method == "stratified":
        raw, covariance = _stratified_samples(weights, quota, num_samples, rng)
    else:
        raw, covariance = _plain_or_antithetic_samples(
            weights, quota, num_samples, batch_size, rng, method
        )

    estimate, standard_error, ci_low, ci_high = _normalize_with_uncertainty(
        raw, covariance, confidence_level
    )
    if not return_result:
        return estimate

    return MonteCarloResult(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        raw_estimate=raw,
        raw_standard_error=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        num_samples=num_samples,
        confidence_level=confidence_level,
        method=method,
    )

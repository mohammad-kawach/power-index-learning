import unittest

import numpy as np

from src.banzhaf import (
    exact_banzhaf,
    exact_shapley_shubik,
    monte_carlo_banzhaf,
)


class MonteCarloBanzhafTests(unittest.TestCase):
    def test_approximates_exact_banzhaf_for_small_game(self):
        weights = np.array([4, 2, 7, 1, 5])

        exact = exact_banzhaf(weights, quota=10)
        approximate = monte_carlo_banzhaf(
            weights,
            quota=10,
            num_samples=10_000,
            seed=42,
        )

        np.testing.assert_allclose(approximate, exact, atol=0.01)

    def test_seed_is_reproducible_across_batch_sizes(self):
        weights = np.arange(1, 21)

        first = monte_carlo_banzhaf(
            weights,
            quota=126,
            num_samples=2_000,
            seed=7,
            batch_size=137,
        )
        second = monte_carlo_banzhaf(
            weights,
            quota=126,
            num_samples=2_000,
            seed=7,
            batch_size=2_000,
        )

        np.testing.assert_array_equal(first, second)

    def test_supports_large_games_and_returns_a_distribution(self):
        estimate = monte_carlo_banzhaf(
            np.arange(1, 51),
            quota=765,
            num_samples=5_000,
            seed=123,
            batch_size=1_000,
        )

        self.assertEqual(estimate.shape, (50,))
        self.assertTrue(np.all(estimate >= 0))
        self.assertAlmostEqual(float(estimate.sum()), 1.0)

    def test_returns_zeros_when_no_agent_can_be_critical(self):
        estimate = monte_carlo_banzhaf(
            np.array([1, 2, 3]),
            quota=100,
            num_samples=100,
            seed=1,
        )

        np.testing.assert_array_equal(estimate, np.zeros(3))

    def test_confidence_interval_contains_estimate(self):
        result = monte_carlo_banzhaf(
            [4, 2, 7, 1, 5],
            quota=10,
            num_samples=5_000,
            seed=9,
            return_result=True,
        )
        self.assertEqual(result.method, "plain")
        self.assertTrue(np.all(result.ci_low <= result.estimate))
        self.assertTrue(np.all(result.estimate <= result.ci_high))
        self.assertTrue(np.all(result.standard_error >= 0))

    def test_variance_reduction_methods_approximate_exact(self):
        weights = np.array([4, 2, 7, 1, 5])
        exact = exact_banzhaf(weights, quota=10)
        for method in ("antithetic", "stratified"):
            with self.subTest(method=method):
                result = monte_carlo_banzhaf(
                    weights,
                    quota=10,
                    num_samples=10_000,
                    seed=11,
                    method=method,
                    return_result=True,
                )
                np.testing.assert_allclose(result.estimate, exact, atol=0.015)

    def test_exact_indices_support_more_agents(self):
        weights = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        banzhaf = exact_banzhaf(weights, quota=5)
        shapley = exact_shapley_shubik(weights, quota=5)
        np.testing.assert_allclose(banzhaf, np.full(8, 1 / 8))
        np.testing.assert_allclose(shapley, np.full(8, 1 / 8))

    def test_shapley_shubik_known_game(self):
        # In [2; 2, 1, 1], the large voter is pivotal in 2/3 of permutations.
        np.testing.assert_allclose(
            exact_shapley_shubik([2, 1, 1], quota=2),
            [2 / 3, 1 / 6, 1 / 6],
        )

    def test_rejects_invalid_sampling_arguments(self):
        invalid_arguments = [
            {"num_samples": 0},
            {"num_samples": 1.5},
            {"batch_size": 0},
            {"batch_size": 1.5},
            {"confidence_level": 1.0},
            {"method": "unknown"},
        ]

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    monte_carlo_banzhaf([1, 2], quota=2, **arguments)

    def test_rejects_invalid_game_arguments(self):
        invalid_games = [
            ([], 1),
            ([[1, 2]], 1),
            ([1, -2], 1),
            ([1, np.inf], 1),
            ([1, 2], np.inf),
            ([1, 2], "not-a-quota"),
        ]

        for weights, quota in invalid_games:
            with self.subTest(weights=weights, quota=quota):
                with self.assertRaises(ValueError):
                    monte_carlo_banzhaf(weights, quota, num_samples=10)


if __name__ == "__main__":
    unittest.main()

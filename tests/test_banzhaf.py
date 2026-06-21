import unittest

import numpy as np

from src.banzhaf import exact_banzhaf, monte_carlo_banzhaf


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

    def test_rejects_invalid_sampling_arguments(self):
        invalid_arguments = [
            {"num_samples": 0},
            {"num_samples": 1.5},
            {"batch_size": 0},
            {"batch_size": 1.5},
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

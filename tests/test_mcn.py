import unittest

import numpy as np

from src.features import create_features_for_one_mcn, create_mcn_feature_matrix
from src.mcn import (
    coalition_value,
    exact_power_indices,
    generate_dataset,
    generate_random_rules,
    marginal_contribution,
    monte_carlo_power_indices,
)


class MarginalContributionNetworkTests(unittest.TestCase):
    def test_paper_rule_example_values(self):
        # Rules: {a and b}->3, {a and not c}->1, {b and not c}->2.
        rules = np.array(
            [
                [1, 1, 0, 0, 0, 0, 3],
                [1, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 0, 0, 1, 2],
            ],
            dtype=float,
        )

        self.assertEqual(coalition_value([1, 0, 0], rules), 1)
        self.assertEqual(coalition_value([0, 1, 0], rules), 2)
        self.assertEqual(coalition_value([1, 1, 0], rules), 6)
        self.assertEqual(coalition_value([1, 0, 1], rules), 0)

    def test_banned_agent_can_have_unsigned_influence(self):
        rules = np.array([[1, 0, 0, 1, 1]], dtype=float)

        self.assertEqual(marginal_contribution(rules, [1, 0], 1, absolute=False), -1)
        self.assertEqual(marginal_contribution(rules, [1, 0], 1, absolute=True), 1)

    def test_exact_indices_for_single_required_agent(self):
        rules = np.array([[1, 0, 0, 0, 1]], dtype=float)

        result = exact_power_indices(rules)

        np.testing.assert_allclose(result.banzhaf, [1, 0])
        np.testing.assert_allclose(result.shapley, [1, 0])

    def test_monte_carlo_approximates_exact(self):
        rules = np.array(
            [
                [1, 0, 0, 0, 1],
                [0, 1, 1, 0, 2],
            ],
            dtype=float,
        )
        exact = exact_power_indices(rules)
        approximate = monte_carlo_power_indices(rules, num_samples=5_000, seed=7)

        np.testing.assert_allclose(approximate.banzhaf, exact.banzhaf, atol=0.03)
        np.testing.assert_allclose(approximate.shapley, exact.shapley, atol=0.03)

    def test_generates_3d_dataset_and_flat_features(self):
        dataset = generate_dataset(
            num_games=5,
            num_rules=4,
            num_agents=3,
            seed=1,
            rule_generator="coin_flip",
            value_generator="low_variance",
            label_method="exact",
        )

        self.assertEqual(dataset["rules"].shape, (5, 4, 7))
        self.assertEqual(dataset["banzhaf_targets"].shape, (5, 3))
        self.assertEqual(create_mcn_feature_matrix(dataset["rules"]).shape, (5, 28))
        self.assertEqual(create_features_for_one_mcn(dataset["rules"][0]).shape, (28,))

    def test_rejects_invalid_rule_generator(self):
        rng = np.random.default_rng(1)
        with self.assertRaises(ValueError):
            generate_random_rules(3, 3, rng=rng, rule_generator="unknown")


if __name__ == "__main__":
    unittest.main()

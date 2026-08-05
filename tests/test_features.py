import unittest

import numpy as np
import pandas as pd

from src.features import (
    create_features_for_one_mcn,
    create_mcn_aggregate_features,
    create_feature_matrix,
    infer_num_agents,
    target_columns,
)


class DynamicFeatureTests(unittest.TestCase):
    def test_infers_agent_count_and_feature_width(self):
        dataframe = pd.DataFrame(
            [{**{f"weight_{i}": i + 1 for i in range(8)}, "quota": 20}]
        )
        self.assertEqual(infer_num_agents(dataframe), 8)
        self.assertEqual(create_feature_matrix(dataframe).shape, (1, 26))

    def test_old_banzhaf_target_names_remain_supported(self):
        dataframe = pd.DataFrame({"target_0": [0.5], "target_1": [0.5]})
        self.assertEqual(target_columns(dataframe, "banzhaf"), ["target_0", "target_1"])

    def test_mcn_augmented_features_include_aggregates(self):
        rules = np.array(
            [
                [1, 0, 0, 1, 2],
                [0, 1, 1, 0, 4],
            ],
            dtype=float,
        )

        raw = create_features_for_one_mcn(rules, feature_set="raw")
        aggregates = create_mcn_aggregate_features(rules)
        augmented = create_features_for_one_mcn(rules)

        self.assertEqual(raw.shape, (10,))
        self.assertEqual(aggregates.shape, (30,))
        self.assertEqual(augmented.shape, (40,))
        np.testing.assert_array_equal(augmented[:10], raw)

    def test_rejects_invalid_mcn_feature_set(self):
        rules = np.array([[1, 0, 0, 0, 1]], dtype=float)

        with self.assertRaises(ValueError):
            create_features_for_one_mcn(rules, feature_set="unknown")


if __name__ == "__main__":
    unittest.main()

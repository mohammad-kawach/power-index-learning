import unittest

import numpy as np
import pandas as pd

from src.features import create_feature_matrix, infer_num_agents, target_columns


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


if __name__ == "__main__":
    unittest.main()

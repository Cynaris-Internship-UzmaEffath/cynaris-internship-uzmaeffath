
import unittest
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "model_comparison.csv"


class TestBiasVarianceRegularisation(unittest.TestCase):

    def test_results_file_exists(self):
        self.assertTrue(RESULTS_FILE.exists())

    def test_results_have_model_comparison_columns(self):
        results = pd.read_csv(RESULTS_FILE)

        expected_columns = {
            "model",
            "train_score",
            "test_score",
        }

        self.assertTrue(expected_columns.issubset(results.columns))

    def test_results_include_models(self):
        results = pd.read_csv(RESULTS_FILE)
        models = results["model"].astype(str).str.lower()

        self.assertTrue(
            models.str.contains("decision tree").any()
            or models.str.contains("decisiontree").any()
        )

        self.assertTrue(
            models.str.contains("logistic").any()
        )

    def test_results_file_is_not_empty(self):
        results = pd.read_csv(RESULTS_FILE)
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()

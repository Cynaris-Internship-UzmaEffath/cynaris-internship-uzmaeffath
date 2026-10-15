
# Tests for W3D3 Decision Trees and Random Forests

import unittest
from pathlib import Path
import pandas as pd

folder = Path(__file__).parent
output_dir = folder / "outputs"


class TestW3D3(unittest.TestCase):

    def test_model_comparison(self):
        file_path = output_dir / "model_comparison.csv"
        self.assertTrue(file_path.exists())

        results = pd.read_csv(file_path)

        self.assertEqual(len(results), 4)
        self.assertIn("Testing Accuracy", results.columns)
        self.assertIn("F1-score", results.columns)

    def test_accuracy_values(self):
        results = pd.read_csv(output_dir / "model_comparison.csv")

        self.assertTrue(
            results["Testing Accuracy"].between(0, 1).all()
        )

    def test_decision_tree_images(self):
        self.assertTrue(
            (output_dir / "decision_tree_gini.png").exists()
        )
        self.assertTrue(
            (output_dir / "decision_tree_tuned.png").exists()
        )

    def test_confusion_matrices(self):
        images = list(output_dir.glob("*confusion_matrix.png"))
        self.assertEqual(len(images), 4)


if __name__ == "__main__":
    unittest.main()

"""Tests for the W4D1 model evaluation implementation."""

import sys
import unittest
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Make the W4D1 directory available for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_evaluation import build_model


class TestModelEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataset = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            test_size=0.2,
            random_state=42,
            stratify=dataset.target,
        )

        cls.X_train = X_train
        cls.X_test = X_test
        cls.y_train = y_train
        cls.y_test = y_test

        cls.model = build_model()
        cls.model.fit(X_train, y_train)

    def test_predictions_have_correct_length(self):
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    def test_accuracy_is_reasonable(self):
        predictions = self.model.predict(self.X_test)
        self.assertGreaterEqual(
            accuracy_score(self.y_test, predictions), 0.90
        )

    def test_roc_auc_is_reasonable(self):
        probabilities = self.model.predict_proba(self.X_test)[:, 1]
        score = roc_auc_score(self.y_test, probabilities)
        self.assertGreaterEqual(score, 0.90)
        self.assertLessEqual(score, 1.0)

    def test_stratified_five_fold_splits(self):
        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42
        )
        splits = list(cv.split(self.X_train, self.y_train))
        self.assertEqual(len(splits), 5)

    def test_prediction_probabilities_are_valid(self):
        probabilities = self.model.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape[1], 2)
        self.assertTrue((probabilities >= 0).all())
        self.assertTrue((probabilities <= 1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)

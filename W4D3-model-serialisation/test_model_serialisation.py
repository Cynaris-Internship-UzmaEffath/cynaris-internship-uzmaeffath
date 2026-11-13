
"""Tests for joblib and pickle model serialisation."""

import pickle

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model():
    dataset = load_breast_cancer()
    X_train, X_test, y_train, _ = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model, X_test


def test_joblib_preserves_predictions(tmp_path):
    model, X_test = train_model()
    path = tmp_path / "model.joblib"

    joblib.dump(model, path)
    loaded_model = joblib.load(path)

    np.testing.assert_array_equal(
        model.predict(X_test),
        loaded_model.predict(X_test),
    )


def test_pickle_preserves_predictions(tmp_path):
    model, X_test = train_model()
    path = tmp_path / "model.pkl"

    with path.open("wb") as file:
        pickle.dump(model, file)

    with path.open("rb") as file:
        loaded_model = pickle.load(file)

    np.testing.assert_array_equal(
        model.predict(X_test),
        loaded_model.predict(X_test),
    )


def test_loaded_model_predicts_expected_number_of_rows():
    model, X_test = train_model()
    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})
if __name__ == "__main__":
     main()
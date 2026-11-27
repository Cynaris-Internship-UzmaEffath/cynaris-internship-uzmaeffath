
"""Train, save, reload, and verify a classification model."""

import pickle
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


OUTPUT_DIR = Path(__file__).parent / "outputs"


def main():
    """Train a model and verify both serialisation methods."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load a built-in dataset.
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    # Train the classification model.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    original_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, original_predictions)

    # Save and reload with joblib.
    joblib_path = OUTPUT_DIR / "model_joblib.pkl"
    joblib.dump(model, joblib_path)
    loaded_joblib = joblib.load(joblib_path)

    # Save and reload with pickle.
    pickle_path = OUTPUT_DIR / "model_pickle.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump(model, file)

    with pickle_path.open("rb") as file:
        loaded_pickle = pickle.load(file)

    # Confirm that serialisation preserves predictions.
    joblib_predictions = loaded_joblib.predict(X_test)
    pickle_predictions = loaded_pickle.predict(X_test)

    assert (original_predictions == joblib_predictions).all()
    assert (original_predictions == pickle_predictions).all()

    print(f"Original model accuracy: {accuracy:.4f}")
    print("joblib: save/load successful; predictions match.")
    print("pickle: save/load successful; predictions match.")
    print(f"joblib file size: {joblib_path.stat().st_size:,} bytes")
    print(f"pickle file size: {pickle_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

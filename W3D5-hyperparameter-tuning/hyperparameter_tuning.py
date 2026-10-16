
"""
W3D5: Hyperparameter Tuning
Compare default, GridSearchCV, and RandomizedSearchCV models.
Dataset: Breast Cancer Wisconsin (built into scikit-learn).
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Create an output folder beside this script.
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load the dataset.
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    # Keep the test set separate from hyperparameter tuning.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Scale features for SVM; the pipeline prevents data leakage.
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC()),
    ])

    # Establish a baseline before tuning.
    baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC()),
    ])

    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    results = [{
        "Model": "Default SVM",
        "Best Parameters": "Default settings",
        "CV Accuracy": None,
        "Test Accuracy": accuracy_score(y_test, baseline_predictions),
        "Test F1": f1_score(y_test, baseline_predictions),
        "Search Time (seconds)": 0.0,
    }]

    # Grid search evaluates every specified combination.
    grid_parameters = {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"],
    }

    grid_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=grid_parameters,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    start = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - start

    grid_predictions = grid_search.predict(X_test)

    results.append({
        "Model": "GridSearchCV SVM",
        "Best Parameters": str(grid_search.best_params_),
        "CV Accuracy": grid_search.best_score_,
        "Test Accuracy": accuracy_score(y_test, grid_predictions),
        "Test F1": f1_score(y_test, grid_predictions),
        "Search Time (seconds)": round(grid_time, 3),
    })

    # Random search samples a fixed number of parameter combinations.
    random_parameters = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    }

    random_search = RandomizedSearchCV(
        estimator=svm_pipeline,
        param_distributions=random_parameters,
        n_iter=15,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )

    start = time.time()
    random_search.fit(X_train, y_train)
    random_time = time.time() - start

    random_predictions = random_search.predict(X_test)

    results.append({
        "Model": "RandomizedSearchCV SVM",
        "Best Parameters": str(random_search.best_params_),
        "CV Accuracy": random_search.best_score_,
        "Test Accuracy": accuracy_score(y_test, random_predictions),
        "Test F1": f1_score(y_test, random_predictions),
        "Search Time (seconds)": round(random_time, 3),
    })

    # Save the comparison table.
    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\n=== Hyperparameter Tuning Results ===")
    print(comparison.to_string(index=False))

    print("\n=== Best Grid Search Parameters ===")
    print(grid_search.best_params_)

    print("\n=== Best Random Search Parameters ===")
    print(random_search.best_params_)

    # Display detailed metrics for the best tuned model.
    best_model = (
        grid_search
        if grid_search.best_score_ >= random_search.best_score_
        else random_search
    )
    best_predictions = best_model.predict(X_test)

    print("\n=== Best Tuned Model Classification Report ===")
    print(classification_report(
        y_test,
        best_predictions,
        target_names=dataset.target_names,
    ))

    # Save a confusion matrix for the best tuned model.
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_predictions,
        display_labels=dataset.target_names,
        cmap="Blues",
    )
    plt.title("Best Tuned SVM - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=150,
    )
    plt.close()

    print(f"\nOutput files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

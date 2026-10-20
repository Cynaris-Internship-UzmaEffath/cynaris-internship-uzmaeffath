
"""Compare model complexity, regularisation, and hyperparameter search."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier


OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def compare_model_complexity(X_train, X_test, y_train, y_test):
    """Compare shallow and deep decision trees."""
    results = []

    for depth in [1, 3, None]:
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42,
        )
        model.fit(X_train, y_train)

        results.append(
            {
                "experiment": "model_complexity",
                "model": f"DecisionTree_depth_{depth}",
                "train_score": model.score(X_train, y_train),
                "test_score": model.score(X_test, y_test),
                "parameter": "max_depth",
                "value": str(depth),
            }
        )

    return results


def compare_regularisation(X_train, X_test, y_train, y_test):
    """Compare logistic regression with different regularisation strengths."""
    results = []

    for c_value in [0.01, 1.0, 100.0]:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)

        results.append(
            {
                "experiment": "regularisation",
                "model": "LogisticRegression",
                "train_score": model.score(X_train, y_train),
                "test_score": model.score(X_test, y_test),
                "parameter": "C",
                "value": c_value,
            }
        )

    return results


def compare_search_methods(X_train, y_train):
    """Compare grid and randomised search using cross-validation."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=3000, random_state=42),
            ),
        ]
    )

    parameter_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "classifier__solver": ["liblinear", "lbfgs"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    searches = [
        (
            "GridSearchCV",
            GridSearchCV(
                pipeline,
                parameter_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
            ),
        ),
        (
            "RandomizedSearchCV",
            RandomizedSearchCV(
                pipeline,
                parameter_grid,
                n_iter=5,
                cv=cv,
                scoring="accuracy",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    results = []

    for name, search in searches:
        search.fit(X_train, y_train)
        results.append(
            {
                "experiment": "hyperparameter_search",
                "model": name,
                "train_score": search.best_estimator_.score(X_train, y_train),
                "test_score": None,
                "parameter": "best_params",
                "value": str(search.best_params_),
                "cv_score": search.best_score_,
            }
        )

    return results


def main():
    """Run all experiments and save their results."""
    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        stratify=dataset.target,
        random_state=42,
    )

    results = []
    results.extend(
        compare_model_complexity(X_train, X_test, y_train, y_test)
    )
    results.extend(
        compare_regularisation(X_train, X_test, y_train, y_test)
    )
    results.extend(compare_search_methods(X_train, y_train))

    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nModel comparison results:")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

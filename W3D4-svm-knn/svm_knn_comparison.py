
"""
W3D4: SVM and KNN Comparison
Dataset: Breast Cancer Wisconsin
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    classification_report,
)


OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def main():
    # Load dataset
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Build pipelines with feature scaling
    models = {
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC()),
            ]),
            {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"],
            },
        ),
        "KNN": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
    }

    # Use stratified five-fold cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for name, (pipeline, parameters) in models.items():
        print(f"\nTraining and tuning {name}...")

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
        )

        # Train and tune using training data only
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "Model": name,
            "Best CV Accuracy": search.best_score_,
            "Test Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Best Parameters": str(search.best_params_),
        })

        # Save confusion matrix
        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=dataset.target_names,
            cmap="Blues",
            values_format="d",
        )
        display.ax_.set_title(f"{name} Confusion Matrix")
        display.figure_.tight_layout()
        display.figure_.savefig(
            OUTPUT_DIR / f"{name.lower()}_confusion_matrix.png",
            dpi=150,
        )
        plt.close(display.figure_)

        # Save classification report
        report = classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names,
            zero_division=0,
        )

        report_path = (
            OUTPUT_DIR
            / f"{name.lower()}_classification_report.txt"
        )

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(f"Best parameters:\n{search.best_params_}\n\n")
            file.write(report)

        print("Best parameters:", search.best_params_)
        print(f"Cross-validation accuracy: {search.best_score_:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(report)

    # Save model comparison table
    comparison = pd.DataFrame(results)
    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nFinal Model Comparison")
    print(comparison.to_string(index=False))
    print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

"""
W4D1: Model Evaluation Metrics
Evaluate a classification model using metrics and cross-validation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(classification_report,ConfusionMatrixDisplay,roc_auc_score,roc_curve,)
from sklearn.model_selection import(train_test_split,StratifiedKFold,cross_val_score,learning_curve,)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
# Save all results in the current project folder
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
def build_model():
    """Create a scaled Logistic Regression model."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
def evaluate_model():
    """Train the model and evaluate its performance."""

    # Load the Breast Cancer dataset
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target

    # Split data while preserving class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train the model
    model = build_model()
    model.fit(X_train, y_train)

    # Generate predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate precision, recall, F1-score and accuracy
    report = classification_report(
        y_test,
        y_pred,
        target_names=dataset.target_names,
        output_dict=True,
        zero_division=0,
    )

    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(OUTPUT_DIR / "classification_metrics.csv")

    print("\n--- Classification Report ---")
    print(classification_report(
        y_test,
        y_pred,
        target_names=dataset.target_names,
        zero_division=0,
    ))

    # Calculate ROC-AUC
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"Test ROC-AUC: {auc_score:.4f}")

    # Plot the confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=dataset.target_names,
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()

    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png")
    plt.close()

    # Perform stratified 5-fold cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_scores = cross_val_score(
        build_model(),
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
    )

    print("\n--- Stratified 5-Fold Cross-Validation ---")
    print("Fold ROC-AUC scores:", np.round(cv_scores, 4))
    print(f"Mean ROC-AUC: {cv_scores.mean():.4f}")
    print(f"Standard deviation: {cv_scores.std():.4f}")

    # Generate learning curves to study overfitting and underfitting
    sizes, train_scores, validation_scores = learning_curve(
        build_model(),
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    plt.figure()
    plt.plot(sizes, train_mean, marker="o", label="Training Accuracy")
    plt.plot(
        sizes,
        validation_mean,
        marker="o",
        label="Validation Accuracy",
    )
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curve.png")
    plt.close()

    # Save learning-curve values for analysis
    learning_df = pd.DataFrame({
        "training_examples": sizes,
        "training_accuracy": train_mean,
        "validation_accuracy": validation_mean,
    })
    learning_df.to_csv(OUTPUT_DIR / "learning_curve_results.csv", index=False)

    print("\n--- Learning Curve Analysis ---")
    print(learning_df.round(4).to_string(index=False))

    print("\nEvaluation completed successfully.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    evaluate_model()

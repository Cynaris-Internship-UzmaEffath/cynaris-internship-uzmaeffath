
from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "aclImdb"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TRAIN_PER_CLASS = 2500
TEST_PER_CLASS = 1000


def load_reviews(split, limit_per_class):
    texts = []
    labels = []

    for folder, label in [("neg", 0), ("pos", 1)]:
        review_dir = DATA_DIR / split / folder
        files = sorted(review_dir.glob("*.txt"))

        # Use a reproducible subset to reduce training time.
        import random
        rng = random.Random(RANDOM_STATE)
        rng.shuffle(files)
        files = files[:limit_per_class]

        for file_path in files:
            texts.append(
                file_path.read_text(encoding="utf-8", errors="replace")
            )
            labels.append(label)

    return texts, labels


def main():
    print("Loading IMDb movie reviews...")
    X_train, y_train = load_reviews("train", TRAIN_PER_CLASS)
    X_test, y_test = load_reviews("test", TEST_PER_CLASS)

    print(f"Training reviews: {len(X_train)}")
    print(f"Test reviews: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []

    for name, classifier in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                stop_words="english",
            )),
            ("classifier", classifier),
        ])

        start = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, probabilities)

        print(f"\n{name} — Classification Report")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["negative", "positive"],
            zero_division=0,
        ))
        print(f"ROC-AUC: {roc_auc:.4f}")

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "Training Time (seconds)": round(elapsed, 2),
        })

        # Save each trained pipeline, including its TF-IDF vectorizer.
        model_filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipeline, MODEL_DIR / model_filename)

        # Save a confusion matrix for this model.
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=["Negative", "Positive"],
            cmap="Blues",
            values_format="d",
        )
        plt.title(f"{name} — Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png",
            dpi=150,
        )
        plt.close()

        # Add the ROC curve for this model.
        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            name=name,
        )
        plt.title(f"{name} — ROC Curve")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_roc_curve.png",
            dpi=150,
        )
        plt.close()

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\nModel Comparison")
    print(comparison.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Models saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()

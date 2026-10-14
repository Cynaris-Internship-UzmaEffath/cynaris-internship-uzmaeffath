
# W3D3: Decision Trees and Random Forests

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Create output folder
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Dataset shape:", X.shape)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Evaluate a model
def evaluate_model(name, model):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print(f"\n{name}")
    print("-" * 40)
    print("Training accuracy:",
          round(accuracy_score(y_train, train_predictions), 4))
    print("Testing accuracy:",
          round(accuracy_score(y_test, test_predictions), 4))
    print("Precision:",
          round(precision_score(y_test, test_predictions), 4))
    print("Recall:",
          round(recall_score(y_test, test_predictions), 4))
    print("F1-score:",
          round(f1_score(y_test, test_predictions), 4))

    return test_predictions


# 1. Decision Tree using Gini impurity
tree_gini = DecisionTreeClassifier(
    criterion="gini", random_state=42
)
tree_gini.fit(X_train, y_train)

gini_predictions = evaluate_model("Decision Tree (Gini)", tree_gini)

# Visualise Gini decision tree
plt.figure(figsize=(18, 10))
plot_tree(
    tree_gini,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)
plt.title("Decision Tree using Gini Impurity")
plt.savefig(
    os.path.join(output_dir, "decision_tree_gini.png"),
    dpi=150, bbox_inches="tight"
)
plt.close()


# 2. Decision Tree using Entropy
tree_entropy = DecisionTreeClassifier(
    criterion="entropy", random_state=42
)
tree_entropy.fit(X_train, y_train)

entropy_predictions = evaluate_model(
    "Decision Tree (Entropy)", tree_entropy
)


# 3. Tune Decision Tree to reduce overfitting
tree_tuned = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
tree_tuned.fit(X_train, y_train)

tuned_predictions = evaluate_model("Tuned Decision Tree", tree_tuned)

# Visualise tuned decision tree
plt.figure(figsize=(16, 9))
plot_tree(
    tree_tuned,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    fontsize=7
)
plt.title("Tuned Decision Tree")
plt.savefig(
    os.path.join(output_dir, "decision_tree_tuned.png"),
    dpi=150, bbox_inches="tight"
)
plt.close()


# 4. Random Forest
random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)
random_forest.fit(X_train, y_train)

forest_predictions = evaluate_model("Random Forest", random_forest)


# 5. Compare all models
models = {
    "Decision Tree (Gini)": tree_gini,
    "Decision Tree (Entropy)": tree_entropy,
    "Tuned Decision Tree": tree_tuned,
    "Random Forest": random_forest
}

results = []

for name, model in models.items():
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Training Accuracy": accuracy_score(y_train, train_predictions),
        "Testing Accuracy": accuracy_score(y_test, test_predictions),
        "Precision": precision_score(y_test, test_predictions),
        "Recall": recall_score(y_test, test_predictions),
        "F1-score": f1_score(y_test, test_predictions)
    })

results_df = pd.DataFrame(results).round(4)

print("\nModel Comparison")
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(output_dir, "model_comparison.csv"),
    index=False
)


# 6. Save confusion matrices
for name, model in models.items():
    predictions = model.predict(X_test)

    matrix_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, predictions),
        display_labels=data.target_names
    )

    matrix_display.plot(cmap="Blues", values_format="d")
    plt.title(name + " - Confusion Matrix")

    file_name = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    plt.savefig(
        os.path.join(output_dir, file_name + "_confusion_matrix.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()


# 7. Random Forest classification report
print("\nRandom Forest Classification Report")
print(classification_report(
    y_test,
    forest_predictions,
    target_names=data.target_names
))

print("\nAll tasks completed successfully.")
print("Output files saved in:", output_dir)

# W2D4: Train/Test Split & Cross-Validation
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# Create a small sample dataset
X = np.array([
    [22, 25000],
    [24, 30000],
    [26, 35000],
    [28, 40000],
    [30, 45000],
    [32, 50000],
    [34, 55000],
    [36, 60000],
    [38, 65000],
    [40, 70000],
    [42, 75000],
    [44, 80000],
    [46, 85000],
    [48, 90000],
    [50, 95000],
    [52, 100000],
    [54, 105000],
    [56, 110000],
    [58, 115000],
    [60, 120000]
])
# Target: 0 = lower income group, 1 = higher income group
y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
])
# 1. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Train/Test Split")
print("----------------")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
# 2. StandardScaler + Logistic Regression
# Pipeline prevents data leakage during cross-validation.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
# 3. Train the model
pipeline.fit(X_train, y_train)
train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)
print("\nModel Performance")
print("-----------------")
print(f"Training accuracy: {train_accuracy:.2f}")
print(f"Testing accuracy: {test_accuracy:.2f}")
# 4. K-Fold Cross-Validation
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)
print("\nCross-Validation")
print("----------------")
print(f"Fold scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.2f}")
print(f"Standard deviation: {cv_scores.std():.2f}")
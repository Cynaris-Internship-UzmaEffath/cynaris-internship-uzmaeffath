import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
df = pd.DataFrame({
    "Age": [24, 25, 27, 28, 29, 31, 32, 34, 35, 36, 38, 40],
    "Salary": [55000, 58000, 62000, 72000, 65000, 60000, 68000, 75000, 80000, 82000, 85000, 90000],
    "Target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]})
X = df[["Age", "Salary"]]
y = df["Target"]
print("\nBefore SMOTE:")
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
smote = SMOTE(random_state=42,k_neighbors=2)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE:")
print(y_train_resampled.value_counts())
# W3D5: Hyperparameter Tuning

## Objective

Compare GridSearchCV and RandomizedSearchCV for tuning a Support Vector Machine (SVM) classifier.

## Dataset

Breast Cancer Wisconsin dataset from scikit-learn.

## Methods

- Default SVM baseline
- GridSearchCV with 5-fold cross-validation
- RandomizedSearchCV with 5-fold cross-validation
- Evaluation using accuracy, F1-score, and a classification report

## Results

- Default SVM test accuracy: 98.25%
- GridSearchCV test accuracy: 98.25%
- RandomizedSearchCV test accuracy: 98.25%
- GridSearchCV best cross-validation accuracy: 97.80%
- RandomizedSearchCV best cross-validation accuracy: 98.02%

## Output Files

- `outputs/model_comparison.csv`
- `outputs/confusion_matrix.png`

## Key Learning

GridSearchCV evaluates all specified parameter combinations, while RandomizedSearchCV samples a fixed number of combinations. In this experiment, randomized search completed faster and achieved slightly better cross-validation accuracy, although both methods produced the same test accuracy as the baseline.

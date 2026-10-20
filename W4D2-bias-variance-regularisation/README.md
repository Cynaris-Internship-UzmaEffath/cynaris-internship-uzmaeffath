# W4D2: Bias-Variance Tradeoff & Regularisation

## Objective

The objective of this task is to understand the bias-variance tradeoff, regularisation, and hyperparameter search techniques using Python and scikit-learn.

## Tools and Technologies

- Python 3.11
- scikit-learn
- pandas
- pytest
- GridSearchCV
- RandomizedSearchCV
- Breast Cancer dataset from scikit-learn

## Experiments Performed

### 1. Bias-Variance Tradeoff

Compared decision trees with different maximum depths: 1, 3, and unlimited depth.

The unlimited-depth tree achieved 100% training accuracy but lower test accuracy than the depth-3 tree. This demonstrates how a more complex model can overfit the training data.

### 2. Regularisation

Compared logistic regression with different values of `C`: 0.01, 1.0, and 100.0.

A smaller `C` applies stronger regularisation, while a larger `C` applies weaker regularisation. In the initial experiment, `C = 1.0` achieved the highest test accuracy among the three settings.

### 3. Hyperparameter Search

Compared Grid Search and Randomised Search using five-fold stratified cross-validation.

- Grid Search tests every combination in the specified parameter grid.
- Randomised Search samples a fixed number of parameter combinations.
- Both approaches select parameters using training data and cross-validation.

## Initial Results

| Experiment                     | Training Accuracy | Test Accuracy |
| ------------------------------ | ----------------: | ------------: |
| Decision Tree, depth 1         |            92.31% |        92.11% |
| Decision Tree, depth 3         |            97.58% |        93.86% |
| Decision Tree, unlimited depth |           100.00% |        91.23% |
| Logistic Regression, C = 0.01  |            95.60% |        94.74% |
| Logistic Regression, C = 1.0   |            98.90% |        98.25% |
| Logistic Regression, C = 100.0 |            99.34% |        94.74% |

Initial cross-validation accuracy:

- Grid Search: 98.24%
- Randomised Search: 97.80%

These results are from the initial run. The search scores are mean cross-validation accuracies on the training data, not test accuracies.

## Output

The experiment saves the model comparison results to:

`results/model_comparison.csv`

## Key Learnings

- A model that performs extremely well on training data may not generalise well to unseen data.
- Regularisation helps control model complexity.
- Grid Search is useful for small, defined parameter grids.
- Randomised Search can explore larger parameter spaces with a limited search budget.
- Cross-validation helps compare hyperparameters more reliably.

## Conclusion

This task demonstrated how model complexity and regularisation affect performance and how Grid Search and Randomised Search can be used to select hyperparameters systematically.

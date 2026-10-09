# W2D4: Train/Test Split & Cross-Validation

## OBJECTIVE:

-> Implement train/test splitting and cross-validation while applying feature scaling correctly.

## TASKS COMPLETED:

- Split the datset info into training and testing sets.
- Used `StandardScalar` to standardize numerical features.
- Used `LogisticRegression` as the classification model.
- Created a Scikit-Learn `Pipeline` to prevent data leakage.
- Applied 5-fold cross-validation using `KFold`.
- Evaluated training, testing, and cross-validation accuracy.

## TOOLS AND TECHNOLOGIES:

- Python
- Pandas
- Numpy
- Scikit-Learn
- StandardScalar
- LogisticRegression
- Pipeline
- train_test_split
- KFold
- cross_val_score

## TRAIN/TEST SPLIT:

-> The dataset was dived into:

- 80% training data
- 20% testing data

NOTE: The Split used `random_state=42` for reproducibility and stratification to maintainthe class distribution.

## FEATURE SCALING

-> `StandardScaler` was used to standardize the numerical features.

-> The scaler was placed inside a Scikit-learn `Pipeline` so that scaling is fitted only on the relevant training data during model evaluation.

This helps prevent data leakage.

## CROSS-VALIDATION:

-> 5-fold cross-validation was performed using `KFold`.

-> The model was evaluated across five different train/validation splits.

### RESULTS:

- Training accuracy: 1.00
- Testing accuracy: 1.00
- Cross-validation scores: `[1. 1. 1. 1. 1.]`
- Mean CV accuracy: 1.00
- Standard deviation: 0.00

NOTE: The perfect score is expected because the sample dataset is small and intentionally simple for demonstrating the train/test split and cross-validation workflow.

## KEY LEARNING:

-> Train/test splitting provides a separate dataset for final evaluation, while cross-validation gives a more reliable estimate of model performance across multiple data splits.

-> Using a pipeline with preprocessing and the model helps avoid data leakage during cross-validation.

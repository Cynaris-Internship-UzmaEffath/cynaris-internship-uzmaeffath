# W4D1: Model Evaluation Metrics

## Objective
Evaluate a classification model using precision, recall, F1-score,
ROC-AUC, stratified cross-validation, and learning curves.

## Dataset
Scikit-learn Breast Cancer Wisconsin dataset.

## Model
Logistic Regression with StandardScaler in a pipeline.

## Evaluation Methods
- Stratified train/test split (80/20)
- Precision, recall, F1-score and classification report
- ROC curve and ROC-AUC
- Confusion matrix
- Stratified 5-fold cross-validation
- Learning curves for diagnosing overfitting and underfitting

## Results
- Test accuracy: approximately 98%
- Test ROC-AUC: 0.9954
- Mean 5-fold ROC-AUC: 0.9959
- Cross-validation ROC-AUC standard deviation: 0.0050

## Generated Outputs
- classification_metrics.csv
- confusion_matrix.png
- roc_curve.png
- learning_curve.png
- learning_curve_results.csv

## Run
```powershell
python .\W4D1-model-evaluation-metrics\model_evaluation.py
```

## Tests
```powershell
python -m unittest discover -s .\W4D1-model-evaluation-metrics -p "test_*.py" -v
```

## Self-Review
- [x] Implemented evaluation metrics
- [x] Added stratified cross-validation
- [x] Generated learning curves
- [ ] Automated tests passed
- [ ] Output evidence reviewed
- [ ] Changes committed and pushed
- [ ] Pull request created

# W3D4: SVM and KNN Comparison

## Objective

Implement and compare Support Vector Machine (SVM) and
K-Nearest Neighbors (KNN) classifiers.

## Dataset

Breast Cancer Wisconsin dataset from scikit-learn.

- Total samples: 569
- Features: 30
- Task: Binary classification
- Test size: 20%
- Random state: 42

## Implementation

- StandardScaler for feature scaling
- Pipeline to help prevent data leakage
- Stratified train/test split
- Five-fold stratified cross-validation
- GridSearchCV for hyperparameter tuning
- Accuracy, precision, recall and F1-score evaluation
- Confusion matrices and classification reports

## Results

| Metric           |    SVM |    KNN |
| ---------------- | -----: | -----: |
| Best CV accuracy | 97.58% | 97.14% |
| Test accuracy    | 98.25% | 96.49% |

SVM achieved the higher test accuracy on this split.
These results do not guarantee the same performance on
every dataset or future example.

## Run the project

```bash
python svm_knn_comparison.py
```

## Generated outputs

- outputs/model_comparison.csv
- outputs/svm_confusion_matrix.png
- outputs/knn_confusion_matrix.png
- outputs/svm_classification_report.txt
- outputs/knn_classification_report.txt

## Conclusion

Both classifiers achieved strong results. SVM performed
better on the held-out test set, while KNN also achieved
high classification performance.

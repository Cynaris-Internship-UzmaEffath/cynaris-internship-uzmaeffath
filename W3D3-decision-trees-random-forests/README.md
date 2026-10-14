# W3D3: Decision Trees and Random Forests

## Objective

Build and compare Decision Tree and Random Forest classifiers.
Understand Gini impurity, entropy, and overfitting.

## Dataset

- Breast Cancer Wisconsin dataset from scikit-learn
- Total samples: 569
- Features: 30
- Training samples: 455
- Testing samples: 114

## Models

1. Decision Tree using Gini impurity
2. Decision Tree using entropy
3. Tuned Decision Tree
4. Random Forest

## Evaluation

Models are evaluated using training accuracy, testing accuracy,
precision, recall, and F1-score.

## Results

| Model                   | Training Accuracy | Testing Accuracy |
| ----------------------- | ----------------: | ---------------: |
| Decision Tree (Gini)    |            1.0000 |           0.9123 |
| Decision Tree (Entropy) |            1.0000 |           0.9123 |
| Tuned Decision Tree     |            0.9758 |           0.9211 |
| Random Forest           |            0.9956 |           0.9474 |

Random Forest achieved the highest testing accuracy in this experiment.
Tuning reduced the Decision Tree's training accuracy and improved
its testing accuracy.

## Output Files

- decision_tree_gini.png
- decision_tree_tuned.png
- model_comparison.csv
- Confusion matrix images for all four models

## Run the Program

```bash
python W3D3-decision-trees-random-forests/decision_trees_random_forests.py
```

## Run Tests

```bash
python -m unittest discover -s W3D3-decision-trees-random-forests -p "test_*.py" -v
```

## Conclusion

The experiment shows that controlling tree depth and minimum sample
requirements can help reduce overfitting. Random Forest performed best
on this particular train-test split.

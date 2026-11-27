# IMDb Sentiment Classification: Logistic Regression vs Random Forest

## Overview

This project compares Logistic Regression and Random Forest for binary sentiment classification using the IMDb movie review dataset. It predicts whether a review expresses positive or negative sentiment.

## Dataset

- Dataset: IMDb Large Movie Review Dataset (`aclImdb`)
- Training subset: 5,000 reviews (2,500 positive and 2,500 negative)
- Test subset: 2,000 reviews (1,000 positive and 1,000 negative)
- Random seed: 42

The experiment uses balanced subsets of the original training and test folders to keep training time manageable.

## Approach

1. Load positive and negative IMDb reviews.
2. Convert review text into TF-IDF features using up to 5,000 features and unigrams/bigrams.
3. Train Logistic Regression and Random Forest classifiers.
4. Evaluate both models using accuracy, precision, recall, F1 score, ROC-AUC, and classification reports.
5. Generate confusion matrices, ROC curves, and a CSV comparison.
6. Save the trained pipelines with Joblib.

## Results

| Metric        | Logistic Regression | Random Forest |
| ------------- | ------------------: | ------------: |
| Accuracy      |              86.65% |        83.30% |
| Precision     |              86.11% |        82.71% |
| Recall        |              87.40% |        84.20% |
| F1 Score      |              86.75% |        83.45% |
| ROC-AUC       |              93.90% |        91.56% |
| Training time |        2.36 seconds |  2.67 seconds |

Logistic Regression performed better across all reported evaluation metrics in this experiment.

## Project Structure

- `classification_comparison.py` — training and evaluation code
- `data/aclImdb/` — extracted IMDb dataset (not required to be committed)
- `outputs/model_comparison.csv` — evaluation metrics
- `outputs/` — confusion matrices and ROC curves
- `models/` — saved Joblib pipelines

## Requirements

Python 3.11 or a compatible Python version, scikit-learn, pandas, matplotlib, and joblib.

## Run

From the repository root, activate your virtual environment and run:

```powershell
python "W4D5-classification-capstone\classification_comparison.py"
```

## Limitations

The reported metrics come from balanced subsets, not the entire IMDb dataset. Results may vary with different subsets, feature settings, or model parameters. The extracted dataset and generated model files should be kept out of version control unless specifically required.

## Future Improvements

- Tune hyperparameters and compare additional text features.
- Evaluate on the full test dataset.
- Add a FastAPI endpoint for predicting sentiment from a new review.

# Model Card: Adult Income Classification

## 1. Model Details
- Model: Logistic Regression
- Dataset: UCI Adult Income
- Framework: scikit-learn
- Fairness toolkit: IBM AI Fairness 360 (AIF360)
- Explainability: SHAP LinearExplainer

## 2. Intended Use
This model is for educational analysis of income classification,
fairness auditing, and explainability. It must not be used to make
real hiring, lending, insurance, or eligibility decisions.

## 3. Dataset and Factors
- Records after cleaning: 30,162
- Features: demographic-independent predictors; gender excluded
  from model inputs and retained for fairness auditing.
- Female records: 9,782
- Male records: 20,380
- Target: income above $50K.

## 4. Performance
- Baseline accuracy: 0.8467
- Reweighted accuracy: 0.8400

## 5. Fairness Analysis
- Baseline female positive prediction rate:
  0.0920
- Baseline male positive prediction rate:
  0.2563
- Baseline disparate impact:
  0.3588
- Reweighted disparate impact:
  0.6341

The 0.80 rule is a screening guideline, not a definitive finding
of discrimination. Reweighing may improve some metrics while
affecting others.

## 6. Explainability
SHAP LinearExplainer was used to examine feature contributions.
The summary plot is saved at outputs/shap_summary.png.

## 7. Limitations and Ethical Considerations
The Adult dataset reflects historical US income patterns and is
not representative of India's population or labour market. Income
labels may reflect historical structural inequalities. Excluding
gender does not eliminate proxy discrimination through correlated
features. The model is not validated for real-world decisions.

## 8. Monitoring and Recommendations
Evaluate subgroup performance, data quality, calibration, and
fairness metrics before any deployment. Document changes and
reassess fairness over time.

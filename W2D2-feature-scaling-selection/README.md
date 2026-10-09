# W2D2: Feature Scaling & Selection

## Objective

Applied categorical encoding, feature scaling and feature selection techniques on the W1D3 dataset.

## Tasks Completed

### 1. Categorical Encoding

Applied:

- LabelEncoder
- OneHotEncoder
- OrdinalEncoder

LabelEncoder converts categories into integer labels and is simple for label-based encoding.

OneHotEncoder creates separate binary columns for categories and avoids creating an artificial order between categories.

OrdinalEncoder converts categories into numerical values and is useful when the categories have a meaningful order.

### 2. Feature Scaling

Applied:

- StandardScaler
- MinMaxScaler
- RobustScaler

StandardScaler transforms values based on mean and standard deviation.

MinMaxScaler scales values to a fixed range, normally 0 to 1.

RobustScaler uses median and interquartile range, so it is more suitable when outliers are present.

### 3. Missing Values

Missing numerical values were handled using median imputation before scaling and feature selection.

Missing categorical values were handled using the most frequent value.

### 4. SelectKBest

SelectKBest with `f_classif` was used to identify the top 5 numerical features based on their scores.

The selected features are saved in:

`outputs/top_5_features.csv`

### 5. Output Files

The following outputs were generated:

- `before_scaling.png`
- `standard_scaler.png`
- `minmax_scaler.png`
- `robust_scaler.png`
- `top_5_features.csv`

## Tools Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- PowerShell
- Git/GitHub

## Key Learnings

- OneHotEncoder is useful for nominal categories.
- OrdinalEncoder is suitable when categories have an order.
- StandardScaler can be affected by outliers.
- RobustScaler is more resistant to outliers.
- Feature scaling can improve the performance of ML algorithms.
- SelectKBest helps identify important features.
- Missing values should be handled before applying many ML algorithms.

## W2D1: Feature Engineerinng and Encoding

## OBJECTIVE:

The objective of the task was to appky common feature engineering techniques including categorical encoding,feature scaling and featurre selection

## DATASET:

The dataset used for this task is the raw dataset from W1D3.
Dataset columns are:
-Name
-Salary
-Age
-City
-Department

## 1. LABEL ENCOING:

### LabelEncoder

-> Label encoder was applied to the dept column to convert categorical values into numeric values

### OneHotEncoder

-> OneHotEncoder was applied to city column. it creates separate binary columns for each category

### OrdinalEncoder

-> It is more suitablr to the dept column to demonstrate ordinal encoding
-> more suitable when categories have a meaningful order such as low, medium, and high

## 2. FEATURE SCALING

->Three scaling techniques were applied to Age and Salary:

### StandardScaler

-> StandardScaler transforms the data so that it is centered around zero with a standard deviation of one.
-> It can be affected by outliers because it uses the mean and standard deviation.

### MinMaxScaler

-> MinMaxScaler scales values to a fixed range, normally between 0 and 1.

### RobustScaler

-> RobustScaler uses the median and interquartile range and is more resistant to outliers.

## 3. DISTRIBUTION ANALYSIS:

-> Distribution plots were created before and after applying the scaling techniques.

--> The plots are available in the `outputs` folder.

## 4. SelectKBest

-> SelectKBest with the ANOVA F-test was used to compare the importance of numerical features.
-> The available numerical input features in this small dataset are:

- Age
- Salary

NOTE: Because the dataset contains only a small number of independent input features, selecting five genuine features was not possible without inventing additional features.

## 5. FEATURE LEAKAGE:

-> The target variable was not used as an input feature during the final feature selection process.

NOTE: This prevents feature leakage and ensures that the model does not receive information that would only be available after the prediction.

## Tools Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Git/GitHub

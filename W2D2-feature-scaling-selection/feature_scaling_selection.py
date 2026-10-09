import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# Load dataset

data_path = ".\\W1D3-data-loading-cleaning\\data\\raw_data.csv"
output_path = ".\\W2D2-feature-scaling-selection\\outputs"

df = pd.read_csv(data_path)

print("Dataset:")
print(df)

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# Handle missing values

numeric_columns = df.select_dtypes(include="number").columns
categorical_columns = df.select_dtypes(include="object").columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing values after handling:")
print(df.isnull().sum())


# Encoding

print("\nCategorical columns:")
print(categorical_columns)


# LabelEncoder

label_df = df.copy()

for column in categorical_columns:
    encoder = LabelEncoder()
    label_df[column] = encoder.fit_transform(label_df[column])

print("\nLabelEncoder:")
print(label_df.head())


# OneHotEncoder

onehot_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

onehot_data = onehot_encoder.fit_transform(
    df[categorical_columns]
)

onehot_columns = onehot_encoder.get_feature_names_out(
    categorical_columns
)

onehot_df = pd.DataFrame(
    onehot_data,
    columns=onehot_columns
)

print("\nOneHotEncoder:")
print(onehot_df.head())


# OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

ordinal_data = ordinal_encoder.fit_transform(
    df[categorical_columns]
)

ordinal_df = pd.DataFrame(
    ordinal_data,
    columns=categorical_columns
)

print("\nOrdinalEncoder:")
print(ordinal_df.head())


# Feature Scaling

numeric_data = df[numeric_columns]

print("\nNumerical columns:")
print(numeric_columns)


# StandardScaler

standard_scaler = StandardScaler()

standard_data = standard_scaler.fit_transform(
    numeric_data
)

standard_df = pd.DataFrame(
    standard_data,
    columns=numeric_columns
)

print("\nStandardScaler:")
print(standard_df.head())


# MinMaxScaler

minmax_scaler = MinMaxScaler()

minmax_data = minmax_scaler.fit_transform(
    numeric_data
)

minmax_df = pd.DataFrame(
    minmax_data,
    columns=numeric_columns
)

print("\nMinMaxScaler:")
print(minmax_df.head())


# RobustScaler

robust_scaler = RobustScaler()

robust_data = robust_scaler.fit_transform(
    numeric_data
)

robust_df = pd.DataFrame(
    robust_data,
    columns=numeric_columns
)

print("\nRobustScaler:")
print(robust_df.head())


# Scaling plots

feature = numeric_columns[0]


plt.figure(figsize=(6, 4))
plt.hist(numeric_data[feature])
plt.title("Before Scaling")
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(output_path + "\\before_scaling.png")
plt.close()


plt.figure(figsize=(6, 4))
plt.hist(standard_df[feature])
plt.title("StandardScaler")
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(output_path + "\\standard_scaler.png")
plt.close()


plt.figure(figsize=(6, 4))
plt.hist(minmax_df[feature])
plt.title("MinMaxScaler")
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(output_path + "\\minmax_scaler.png")
plt.close()


plt.figure(figsize=(6, 4))
plt.hist(robust_df[feature])
plt.title("RobustScaler")
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(output_path + "\\robust_scaler.png")
plt.close()


# SelectKBest

print("\nSelectKBest:")

X = df[numeric_columns[:-1]]
y = df[numeric_columns[-1]]

k = min(5, X.shape[1])

selector = SelectKBest(
    score_func=f_classif,
    k=k
)

selector.fit(X, y)

scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
})

scores = scores.sort_values(
    by="Score",
    ascending=False
)

top_features = scores.head(k)

print("\nFeature scores:")
print(scores)

print("\nTop 5 features:")
print(top_features)


top_features.to_csv(
    output_path + "\\top_5_features.csv",
    index=False
)


print("\nW2D2 completed successfully.")
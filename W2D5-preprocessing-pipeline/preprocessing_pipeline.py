from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# File paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "titanic.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Load the dataset
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully")
print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
# Check column information
print("\nColumn information:")
print(df.info())
# Check missing values
print("\nMissing values:")
print(df.isnull().sum())
# Save basic EDA results
df.describe(include="all").transpose().to_csv(OUTPUT_DIR / "eda_summary.csv")
df.isnull().sum().to_csv(OUTPUT_DIR / "missing_values.csv")
# Plot survival distribution
df["Survived"].value_counts().sort_index().plot(kind="bar")
plt.title("Titanic Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "survival_distribution.png")
plt.close()
# Remove rows where the target is missing
df = df.dropna(subset=["Survived"]).copy()
# Select the features needed for preprocessing
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]
X = df[features].copy()
y = df["Survived"].astype(int).copy()
# Separate numerical and categorical columns
numeric_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_columns = ["Sex", "Embarked"]
# Handle missing values and scale numerical data
numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler()),])
# Fill missing categories and encode them
categorical_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore")),])
# Apply preprocessing to the correct columns
preprocessor = ColumnTransformer(transformers=[("numeric", numeric_pipeline, numeric_columns),("categorical", categorical_pipeline, categorical_columns),])
# Transform the dataset
processed_data = preprocessor.fit_transform(X)
# Convert the output to a regular array if needed
if hasattr(processed_data, "toarray"):
    processed_data = processed_data.toarray()
# Get the column names after encoding
column_names = preprocessor.get_feature_names_out()
processed_df = pd.DataFrame(processed_data,columns=column_names,)
# Save the processed features and target separately
processed_df.to_csv( OUTPUT_DIR / "titanic_features.csv",index=False,)
y.reset_index(drop=True).to_frame().to_csv(OUTPUT_DIR / "titanic_target.csv",index=False,)
# Save one combined file as well
combined_df = processed_df.copy()
combined_df["Survived"] = y.reset_index(drop=True)
combined_df.to_csv(OUTPUT_DIR / "titanic_preprocessed.csv",index=False,)
# Basic checks after preprocessing
assert len(processed_df) == len(y)
assert processed_df.isnull().sum().sum() == 0
assert processed_df.shape[1] > 0
assert processed_df.select_dtypes(include="number").shape[1] == (processed_df.shape[1])
print("\nPreprocessing completed successfully")
print("Original feature shape:", X.shape)
print("Processed feature shape:", processed_df.shape)
print("Target shape:", y.shape)
print("\nMissing values after preprocessing:")
print(processed_df.isnull().sum().sum())
print("\nFiles saved in the outputs folder:")
print("- eda_summary.csv")
print("- missing_values.csv")
print("- survival_distribution.png")
print("- titanic_features.csv")
print("- titanic_target.csv")
print("- titanic_preprocessed.csv")

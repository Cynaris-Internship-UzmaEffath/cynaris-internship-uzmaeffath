# W2D5: End-to-End Preprocessing Pipeline

## Objective

The objective of this project is to apply Week 2 preprocessing techniques to the Titanic dataset and prepare the data for machine learning.

## Tasks Completed

- Loaded and explored the Titanic dataset.
- Checked missing values and generated EDA reports.
- Created a survival distribution plot.
- Handled missing numerical values using median imputation.
- Handled missing categorical values using most-frequent imputation.
- Encoded categorical features using One-Hot Encoding.
- Scaled numerical features using StandardScaler.
- Exported the processed features and target.
- Added basic validation checks and a test script.

## Tools Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- PowerShell
- Git and GitHub

## Project Structure

- `data/titanic.csv` — Input dataset
- `outputs/` — EDA reports, plot and processed datasets
- `preprocessing_pipeline.py` — Main preprocessing script
- `test_preprocessing.py` — Validation tests

## Results

- Input dataset: 891 rows
- Original input features: 7
- Processed numerical features: 10
- Missing values after preprocessing: 0

## How to Run

Run the following commands from the project folder:

----> powershell
python .\preprocessing_pipeline.py
python .\test_preprocessing.py

## Learning Outcome

This project helped me practise EDA, missing-value handling, categorical encoding, feature scaling and exporting ML-ready data using Scikit-learn pipelines.

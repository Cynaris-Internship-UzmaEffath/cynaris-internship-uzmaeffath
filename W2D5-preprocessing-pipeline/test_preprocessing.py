
# Tests for W2D5 preprocessing pipeline

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def test_output_files_exist():
    """Check that all expected output files were created."""
    expected_files = [
        "eda_summary.csv",
        "missing_values.csv",
        "survival_distribution.png",
        "titanic_features.csv",
        "titanic_target.csv",
        "titanic_preprocessed.csv",
    ]

    for file_name in expected_files:
        assert (OUTPUT_DIR / file_name).exists(), (
            f"Missing output file: {file_name}"
        )


def test_processed_data():
    """Check the processed features and target."""
    features = pd.read_csv(OUTPUT_DIR / "titanic_features.csv")
    target = pd.read_csv(OUTPUT_DIR / "titanic_target.csv")
    combined = pd.read_csv(
        OUTPUT_DIR / "titanic_preprocessed.csv"
    )

    assert len(features) == 891
    assert len(target) == 891
    assert len(combined) == 891

    assert features.shape[1] == 10
    assert "Survived" in combined.columns

    assert features.isnull().sum().sum() == 0
    assert features.select_dtypes(include="number").shape[1] == 10

    assert set(target["Survived"].unique()).issubset({0, 1})


if __name__ == "__main__":
    test_output_files_exist()
    test_processed_data()
    print("All W2D5 tests passed!")

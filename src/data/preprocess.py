
"""
Phase 4: Data Cleaning

Project: Student Performance & At-Risk Prediction

This script:
1. Loads raw data.
2. Validates schema.
3. Converts date columns.
4. Checks missing values.
5. Checks duplicates.
6. Validates numerical ranges.
7. Validates target consistency.
8. Saves cleaned data.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------
# 1. PROJECT PATHS
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_performance.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "student_performance_cleaned.csv"
)


# -------------------------------------------------
# 2. DATASET CONFIGURATION
# -------------------------------------------------

REQUIRED_COLUMNS = [
    "student_id",
    "prediction_period",
    "prediction_date",
    "next_assessment_date",
    "attendance_rate",
    "assignment_completion_rate",
    "previous_score_1",
    "previous_score_2",
    "previous_score_3",
    "next_assessment_score",
    "at_risk_next_assessment"
]

PERCENTAGE_COLUMNS = [
    "attendance_rate",
    "assignment_completion_rate",
    "previous_score_1",
    "previous_score_2",
    "previous_score_3",
    "next_assessment_score"
]

DATE_COLUMNS = [
    "prediction_date",
    "next_assessment_date"
]

TARGET_COLUMN = "at_risk_next_assessment"

RISK_THRESHOLD = 40


# -------------------------------------------------
# 3. LOAD DATASET
# -------------------------------------------------

def load_dataset():

    if not RAW_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print("\nRaw dataset loaded successfully.")

    print("Original shape:", df.shape)

    return df


# -------------------------------------------------
# 4. CHECK REQUIRED COLUMNS
# -------------------------------------------------

def validate_columns(df):

    missing_columns = (
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("\nColumn validation passed.")


# -------------------------------------------------
# 5. CONVERT DATA TYPES
# -------------------------------------------------

def convert_data_types(df):

    df = df.copy()

    # Convert dates.
    for column in DATE_COLUMNS:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    # Convert numerical columns.
    numeric_columns = [
        "prediction_period",
        *PERCENTAGE_COLUMNS,
        TARGET_COLUMN
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Validate student IDs.
    df["student_id"] = (
        df["student_id"].astype("string")
    )

    print("\nData type conversion completed.")

    return df


# -------------------------------------------------
# 6. HANDLE MISSING VALUES
# -------------------------------------------------

def validate_missing_values(df):

    missing_counts = df.isnull().sum()

    print("\nMissing value analysis:")

    print(missing_counts)

    if missing_counts.sum() > 0:

        raise ValueError(
            "Missing values detected. "
            "Investigate and handle them before proceeding."
        )

    print("\nNo missing values detected.")

    return df


# -------------------------------------------------
# 7. HANDLE DUPLICATE RECORDS
# -------------------------------------------------

def handle_duplicates(df):

    duplicate_rows = df.duplicated().sum()

    duplicate_student_period = df.duplicated(
        subset=["student_id", "prediction_period"]
    ).sum()

    print("\nDuplicate analysis:")

    print("Completely duplicated rows:", duplicate_rows)

    print(
        "Duplicate student-period records:",
        duplicate_student_period
    )

    if duplicate_student_period > 0:

        raise ValueError(
            "Duplicate student-period records found. "
            "Investigate before removing any records."
        )

    # Remove only exact duplicate rows.
    df = df.drop_duplicates().copy()

    print("Duplicate validation completed.")

    return df


# -------------------------------------------------
# 8. VALIDATE NUMERICAL RANGES
# -------------------------------------------------

def validate_numerical_ranges(df):

    for column in PERCENTAGE_COLUMNS:

        invalid = ~df[column].between(0, 100)

        if invalid.any():

            raise ValueError(
                f"Invalid values detected in {column}"
            )

    if (df["prediction_period"] <= 0).any():

        raise ValueError(
            "Prediction periods must be positive."
        )

    if not df[TARGET_COLUMN].isin([0, 1]).all():

        raise ValueError(
            "Target variable must contain only 0 and 1."
        )

    print("\nNumerical range validation passed.")

    return df


# -------------------------------------------------
# 9. VALIDATE DATE LOGIC
# -------------------------------------------------

def validate_dates(df):

    if df[DATE_COLUMNS].isnull().any().any():

        raise ValueError(
            "Invalid dates found in dataset."
        )

    invalid_dates = (
        df["next_assessment_date"]
        <= df["prediction_date"]
    )

    if invalid_dates.any():

        raise ValueError(
            "Upcoming assessment must occur after prediction date."
        )

    print("\nDate validation passed.")

    return df


# -------------------------------------------------
# 10. VALIDATE TARGET CONSISTENCY
# -------------------------------------------------

def validate_target(df):

    expected_target = (
        df["next_assessment_score"] < RISK_THRESHOLD
    ).astype("int64")

    if not (
        expected_target.to_numpy()
        == df[TARGET_COLUMN].to_numpy()
    ).all():

        raise ValueError(
            "Target labels do not match future assessment scores."
        )

    print("\nTarget consistency validation passed.")

    return df


# -------------------------------------------------
# 11. SORT DATA CHRONOLOGICALLY
# -------------------------------------------------

def sort_dataset(df):

    df = df.sort_values(
        by=[
            "prediction_period",
            "student_id"
        ]
    ).reset_index(drop=True)

    print("\nDataset sorted chronologically.")

    return df


# -------------------------------------------------
# 12. MAIN PREPROCESSING PIPELINE
# -------------------------------------------------

def preprocess_data():

    print("\nStarting data cleaning pipeline...")

    df = load_dataset()

    validate_columns(df)

    df = convert_data_types(df)

    df = validate_missing_values(df)

    df = handle_duplicates(df)

    df = validate_numerical_ranges(df)

    df = validate_dates(df)

    df = validate_target(df)

    df = sort_dataset(df)

    # Save cleaned dataset.
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print("\nData cleaning completed successfully.")

    print("Final shape:", df.shape)

    print("Saved at:", PROCESSED_DATA_PATH)

    return df


# -------------------------------------------------
# 13. SCRIPT ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":

    preprocess_data()

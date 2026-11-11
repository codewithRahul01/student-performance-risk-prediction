
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_performance.csv"
)

START_DATE = pd.Timestamp("2025-01-01")

# Features that will be available to the ML model.
MODEL_INPUT_FEATURES = [
    "attendance_rate",
    "assignment_completion_rate",
    "previous_score_1",
    "previous_score_2",
    "previous_score_3",
]

TARGET_COLUMN = "at_risk_next_assessment"

# A student is considered at risk if the next score is below 40.
AT_RISK_THRESHOLD = 40


def clip_values(values, minimum=0, maximum=100):
    """
    Restrict generated values to a specified range.

    Example:
    -5 becomes 0
    105 becomes 100
    """

    return np.clip(values, minimum, maximum)

def generate_dataset(
    n_students=1000,
    n_prediction_periods=5,
    random_seed=42
):
   
    if n_students <= 0:
        raise ValueError("Number of students must be positive.")

    if n_prediction_periods <= 0:
        raise ValueError("Prediction periods must be positive.")

    # Create a reproducible random number generator.
    rng = np.random.default_rng(random_seed)

    student_ids = np.array([
        f"S{i:05d}"
        for i in range(1, n_students + 1)
    ])

    academic_ability = rng.normal(
        loc=60,
        scale=15,
        size=n_students
    )

    # Generate individual attendance tendencies.
    base_attendance = clip_values(
        74
        + 0.35 * (academic_ability - 60)
        + rng.normal(0, 8, n_students),
        30,
        100
    )

    # Generate individual assignment completion tendencies.
    base_assignment = clip_values(
        70
        + 0.30 * (academic_ability - 60)
        + rng.normal(0, 10, n_students),
        20,
        100
    )

    score_oldest = clip_values(
        academic_ability
        + rng.normal(0, 12, n_students)
    )

    score_middle = clip_values(
        0.75 * academic_ability
        + 0.25 * score_oldest
        + rng.normal(0, 8, n_students)
    )

    score_recent = clip_values(
        0.75 * academic_ability
        + 0.25 * score_middle
        + rng.normal(0, 8, n_students)
    )

    # Maintain historical scores for every student.
    history_scores = [
        score_oldest,
        score_middle,
        score_recent
    ]

    all_records = []

    for period in range(n_prediction_periods):

        prediction_period = period + 1

        prediction_date = (
            START_DATE + pd.DateOffset(months=period)
        )

        # Upcoming assessment is 14 days after prediction.
        next_assessment_date = (
            prediction_date + pd.Timedelta(days=14)
        )

        # Generate current attendance.
        attendance = clip_values(
            base_attendance
            + rng.normal(0, 6, n_students)
            - period * 0.4
        )

        # Generate current assignment completion.
        assignment_completion = clip_values(
            base_assignment
            + rng.normal(0, 8, n_students)
            - period * 0.3
        )

        # Retrieve the three most recent historical scores.
        previous_score_1 = history_scores[-1]
        previous_score_2 = history_scores[-2]
        previous_score_3 = history_scores[-3]

        next_score = clip_values(
            0.50 * previous_score_1
            + 0.20 * previous_score_2
            + 0.10 * previous_score_3
            + 0.12 * attendance
            + 0.08 * assignment_completion
            + rng.normal(0, 7, n_students)
        )

        at_risk = (
            next_score < AT_RISK_THRESHOLD
        ).astype("int8")

        # -------------------------------------------------
        # STEP G: CREATE THE PERIOD DATAFRAME
        # -------------------------------------------------

        period_df = pd.DataFrame({

            "student_id": student_ids,

            "prediction_period": prediction_period,

            "prediction_date": prediction_date,

            "next_assessment_date": next_assessment_date,

            "attendance_rate": np.round(attendance, 2),

            "assignment_completion_rate": np.round(
                assignment_completion, 2
            ),

            "previous_score_1": np.round(
                previous_score_1, 2
            ),

            "previous_score_2": np.round(
                previous_score_2, 2
            ),

            "previous_score_3": np.round(
                previous_score_3, 2
            ),

            "next_assessment_score": np.round(
                next_score, 2
            ),

            "at_risk_next_assessment": at_risk

        })

        all_records.append(period_df)

        # Update historical scores.
        # The upcoming assessment becomes historical
        # information for the next prediction period.

        history_scores.append(next_score)

  
    dataset = pd.concat(
        all_records,
        ignore_index=True
    )

    dataset = dataset.sort_values(
        by=["prediction_period", "student_id"]
    ).reset_index(drop=True)

    return dataset

def validate_dataset(dataset, n_students, n_periods):

    """
    Check that the generated dataset is valid.
    """

    required_columns = [
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

    # Check required columns.
    missing_columns = (
        set(required_columns) - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Check expected record count.
    expected_rows = n_students * n_periods

    if len(dataset) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"but found {len(dataset)}."
        )

    # Check missing values.
    if dataset[required_columns].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    # Check duplicate student-period records.
    if dataset.duplicated(
        subset=["student_id", "prediction_period"]
    ).any():

        raise ValueError(
            "Duplicate student-period records found."
        )

    # Check percentage and score ranges.
    numerical_columns = [
        "attendance_rate",
        "assignment_completion_rate",
        "previous_score_1",
        "previous_score_2",
        "previous_score_3",
        "next_assessment_score"
    ]

    for column in numerical_columns:

        if not dataset[column].between(0, 100).all():

            raise ValueError(
                f"Invalid values detected in {column}."
            )

    # Check target values.
    if not dataset[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    # Check target-label consistency.
    expected_target = (
        dataset["next_assessment_score"] < AT_RISK_THRESHOLD
    ).astype("int8")

    if not np.array_equal(
        dataset[TARGET_COLUMN].to_numpy(),
        expected_target.to_numpy()
    ):

        raise ValueError(
            "Target labels do not match assessment scores."
        )

    print("\nAll data validation checks passed!")



def main():

    parser = argparse.ArgumentParser(
        description="Generate synthetic student performance data."
    )

    parser.add_argument(
        "--students",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--periods",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT
    )

    args = parser.parse_args()

    print("\nGenerating student performance dataset...")

    dataset = generate_dataset(
        n_students=args.students,
        n_prediction_periods=args.periods,
        random_seed=args.seed
    )

    validate_dataset(
        dataset,
        n_students=args.students,
        n_periods=args.periods
    )

    # Create output directory if it doesn't exist.
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save dataset.
    dataset.to_csv(
        args.output,
        index=False
    )

 
    print("\nDataset successfully generated!")

    print(f"\nTotal records: {len(dataset)}")

    print(
        f"Unique students: {dataset['student_id'].nunique()}"
    )

    print(
        f"Prediction periods: "
        f"{dataset['prediction_period'].nunique()}"
    )

    print(f"Total columns: {len(dataset.columns)}")

    print(f"\nSaved at: {args.output}")

    print("\nTarget distribution:")

    print(
        dataset[TARGET_COLUMN]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(index={
            0: "Not at risk",
            1: "At risk"
        })
    )

    print("\nTarget percentages:")

    print(
        (
            dataset[TARGET_COLUMN]
            .value_counts(normalize=True)
            .reindex([0, 1], fill_value=0)
            * 100
        ).round(2)
    )

    print("\nFirst 5 records:")

    print(dataset.head().to_string(index=False))


if __name__ == "__main__":
    main()

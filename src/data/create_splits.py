
"""
Phase 4: Chronological Train / Validation / Test Split

Training:
Periods 1, 2, 3

Validation:
Period 4

Testing:
Period 5

No random shuffling.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------
# 1. PATH CONFIGURATION
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "student_performance_features.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)


# -------------------------------------------------
# 2. SPLIT CONFIGURATION
# -------------------------------------------------

TRAIN_PERIODS = [1, 2, 3]

VALIDATION_PERIODS = [4]

TEST_PERIODS = [5]


# -------------------------------------------------
# 3. LOAD DATA
# -------------------------------------------------

def load_data():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "Engineered dataset not found. "
            "Run build_features.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    return df


# -------------------------------------------------
# 4. CREATE CHRONOLOGICAL SPLITS
# -------------------------------------------------

def create_splits(df):

    train_df = df[
        df["prediction_period"].isin(TRAIN_PERIODS)
    ].copy()

    validation_df = df[
        df["prediction_period"].isin(VALIDATION_PERIODS)
    ].copy()

    test_df = df[
        df["prediction_period"].isin(TEST_PERIODS)
    ].copy()

    return train_df, validation_df, test_df


# -------------------------------------------------
# 5. VALIDATE SPLITS
# -------------------------------------------------

def validate_splits(
    train_df,
    validation_df,
    test_df
):

    # Expected sizes.
    assert len(train_df) == 3000

    assert len(validation_df) == 1000

    assert len(test_df) == 1000

    # Chronological order.
    assert (
        train_df["prediction_period"].max()
        <
        validation_df["prediction_period"].min()
    )

    assert (
        validation_df["prediction_period"].max()
        <
        test_df["prediction_period"].min()
    )

    # Ensure no missing periods.
    assert set(
        train_df["prediction_period"].unique()
    ) == {1, 2, 3}

    assert set(
        validation_df["prediction_period"].unique()
    ) == {4}

    assert set(
        test_df["prediction_period"].unique()
    ) == {5}

    # Ensure every student is present in every split.
    assert (
        train_df["student_id"].nunique() == 1000
    )

    assert (
        validation_df["student_id"].nunique() == 1000
    )

    assert (
        test_df["student_id"].nunique() == 1000
    )

    print("\nAll chronological split checks passed!")


# -------------------------------------------------
# 6. SAVE SPLITS
# -------------------------------------------------

def save_splits(
    train_df,
    validation_df,
    test_df
):

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        SPLIT_DIR / "train.csv",
        index=False
    )

    validation_df.to_csv(
        SPLIT_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        SPLIT_DIR / "test.csv",
        index=False
    )

    print("\nSplit files saved successfully.")

    print("Train:", len(train_df))

    print("Validation:", len(validation_df))

    print("Test:", len(test_df))


# -------------------------------------------------
# 7. MAIN
# -------------------------------------------------

def main():

    print("\nCreating chronological splits...")

    df = load_data()

    train_df, validation_df, test_df = (
        create_splits(df)
    )

    validate_splits(
        train_df,
        validation_df,
        test_df
    )

    save_splits(
        train_df,
        validation_df,
        test_df
    )


if __name__ == "__main__":

    main()

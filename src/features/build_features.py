
"""
Phase 4: Feature Engineering

Project: Student Performance & At-Risk Prediction

Creates additional features from historical academic data.

No future outcome is used to construct predictive features.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------
# 1. PROJECT PATHS
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "student_performance_cleaned.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "student_performance_features.csv"
)


# -------------------------------------------------
# 2. FEATURE CONFIGURATION
# -------------------------------------------------

HISTORICAL_SCORE_COLUMNS = [
    "previous_score_1",
    "previous_score_2",
    "previous_score_3"
]

MODEL_FEATURES = [
    "attendance_rate",
    "assignment_completion_rate",
    "previous_score_1",
    "previous_score_2",
    "previous_score_3",
    "average_previous_score",
    "score_trend",
    "recent_score_change",
    "score_volatility"
]

TARGET_COLUMN = "at_risk_next_assessment"


# -------------------------------------------------
# 3. LOAD CLEANED DATA
# -------------------------------------------------

def load_cleaned_data():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "Cleaned dataset not found. "
            "Run preprocess.py first."
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=[
            "prediction_date",
            "next_assessment_date"
        ]
    )

    print("Cleaned dataset loaded:", df.shape)

    return df


# -------------------------------------------------
# 4. CREATE AVERAGE HISTORICAL SCORE
# -------------------------------------------------

def create_average_score(df):

    df = df.copy()

    df["average_previous_score"] = (
        df[HISTORICAL_SCORE_COLUMNS].mean(axis=1)
    )

    return df


# -------------------------------------------------
# 5. CREATE ACADEMIC PERFORMANCE TREND
# -------------------------------------------------

def create_score_trend(df):

    df = df.copy()

    df["score_trend"] = (
        df["previous_score_1"]
        - df["previous_score_3"]
    ) / 2

    return df


# -------------------------------------------------
# 6. CREATE RECENT SCORE CHANGE
# -------------------------------------------------

def create_recent_score_change(df):

    df = df.copy()

    df["recent_score_change"] = (
        df["previous_score_1"]
        - df["previous_score_2"]
    )

    return df


# -------------------------------------------------
# 7. CREATE SCORE VOLATILITY
# -------------------------------------------------

def create_score_volatility(df):

    df = df.copy()

    df["score_volatility"] = (
        df[HISTORICAL_SCORE_COLUMNS]
        .std(axis=1, ddof=0)
    )

    return df


# -------------------------------------------------
# 8. VALIDATE ENGINEERED FEATURES
# -------------------------------------------------

def validate_features(df):

    # Ensure all expected features exist.
    missing_features = (
        set(MODEL_FEATURES) - set(df.columns)
    )

    if missing_features:

        raise ValueError(
            f"Missing engineered features: {missing_features}"
        )

    # Check missing feature values.
    if df[MODEL_FEATURES].isnull().any().any():

        raise ValueError(
            "Missing values detected in engineered features."
        )

    # Check infinite values.
    if not np.isfinite(
        df[MODEL_FEATURES].to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Infinite values detected in features."
        )

    # Ensure future outcomes are not model inputs.
    forbidden_features = [
        "next_assessment_score",
        TARGET_COLUMN,
        "student_id"
    ]

    overlap = (
        set(MODEL_FEATURES) & set(forbidden_features)
    )

    if overlap:

        raise ValueError(
            f"Leakage detected: {overlap}"
        )

    print("\nFeature validation passed.")

    return df


# -------------------------------------------------
# 9. BUILD COMPLETE FEATURE PIPELINE
# -------------------------------------------------

def build_features():

    print("\nStarting feature engineering...")

    df = load_cleaned_data()

    df = create_average_score(df)

    df = create_score_trend(df)

    df = create_recent_score_change(df)

    df = create_score_volatility(df)

    df = validate_features(df)

    # Save engineered dataset.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFeature engineering completed!")

    print("Final dataset shape:", df.shape)

    print("\nNew features created:")

    print([
        "average_previous_score",
        "score_trend",
        "recent_score_change",
        "score_volatility"
    ])

    print("\nSaved at:", OUTPUT_PATH)

    return df


# -------------------------------------------------
# 10. SCRIPT ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":

    build_features()

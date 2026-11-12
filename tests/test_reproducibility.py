
from pathlib import Path

import numpy as np

from src.inference.predictor import (
    StudentRiskPredictor,
    MODEL_PATH,
    EXPECTED_FEATURES
)


def test_model_artifact_exists():

    assert MODEL_PATH.exists()


def test_model_feature_configuration():

    predictor = StudentRiskPredictor()

    assert predictor.feature_columns == EXPECTED_FEATURES


def test_prediction_is_reproducible():

    predictor = StudentRiskPredictor()

    inputs = {
        "attendance_rate": 75,
        "assignment_completion_rate": 80,
        "previous_score_1": 45,
        "previous_score_2": 48,
        "previous_score_3": 50
    }

    result_1 = predictor.predict(**inputs)
    result_2 = predictor.predict(**inputs)

    assert result_1["risk_probability"] == result_2["risk_probability"]

    assert result_1["prediction"] == result_2["prediction"]


def test_prediction_probability_is_valid():

    predictor = StudentRiskPredictor()

    result = predictor.predict(
        attendance_rate=75,
        assignment_completion_rate=80,
        previous_score_1=45,
        previous_score_2=48,
        previous_score_3=50
    )

    probability = result["risk_probability"]

    assert np.isfinite(probability)

    assert 0 <= probability <= 1


def test_no_target_leakage_in_features():

    forbidden_features = {
        "student_id",
        "next_assessment_score",
        "at_risk_next_assessment",
        "prediction_date",
        "next_assessment_date"
    }

    assert not forbidden_features.intersection(
        EXPECTED_FEATURES
    )


import pytest
import numpy as np

from src.inference.predictor import (
    StudentRiskPredictor,
    create_student_features
)


# -------------------------------------------------
# TEST 1: FEATURE ENGINEERING
# -------------------------------------------------

def test_feature_calculation():

    features = create_student_features(
        attendance_rate=75,
        assignment_completion_rate=80,
        previous_score_1=45,
        previous_score_2=48,
        previous_score_3=50
    )

    assert features["average_previous_score"].iloc[0] == pytest.approx(
        47.6666667
    )

    assert features["score_trend"].iloc[0] == -2.5

    assert features["recent_score_change"].iloc[0] == -3

    assert features["score_volatility"].iloc[0] == pytest.approx(
        2.0548047
    )


# -------------------------------------------------
# TEST 2: INVALID ATTENDANCE
# -------------------------------------------------

@pytest.mark.parametrize("attendance", [-1, 101])

def test_invalid_attendance(attendance):

    with pytest.raises(ValueError):

        create_student_features(
            attendance_rate=attendance,
            assignment_completion_rate=80,
            previous_score_1=45,
            previous_score_2=48,
            previous_score_3=50
        )


# -------------------------------------------------
# TEST 3: INVALID ASSIGNMENT COMPLETION
# -------------------------------------------------

@pytest.mark.parametrize("assignment", [-5, 110])

def test_invalid_assignment(assignment):

    with pytest.raises(ValueError):

        create_student_features(
            attendance_rate=75,
            assignment_completion_rate=assignment,
            previous_score_1=45,
            previous_score_2=48,
            previous_score_3=50
        )


# -------------------------------------------------
# TEST 4: INVALID ASSESSMENT SCORE
# -------------------------------------------------

@pytest.mark.parametrize("score", [-10, 105])

def test_invalid_assessment_score(score):

    with pytest.raises(ValueError):

        create_student_features(
            attendance_rate=75,
            assignment_completion_rate=80,
            previous_score_1=score,
            previous_score_2=48,
            previous_score_3=50
        )


# -------------------------------------------------
# TEST 5: NON-FINITE INPUTS
# -------------------------------------------------

@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf]
)

def test_non_finite_inputs(invalid_value):

    with pytest.raises(ValueError):

        create_student_features(
            attendance_rate=75,
            assignment_completion_rate=80,
            previous_score_1=invalid_value,
            previous_score_2=48,
            previous_score_3=50
        )


# -------------------------------------------------
# TEST 6: MODEL PREDICTION
# -------------------------------------------------

def test_model_prediction():

    predictor = StudentRiskPredictor()

    result = predictor.predict(
        attendance_rate=75,
        assignment_completion_rate=80,
        previous_score_1=45,
        previous_score_2=48,
        previous_score_3=50
    )

    assert 0 <= result["risk_probability"] <= 1

    assert result["prediction"] in [0, 1]

    assert result["risk_band"] in [
        "Low",
        "Moderate",
        "High"
    ]

    assert len(result["features"]) == 9


# -------------------------------------------------
# TEST 7: MODEL THRESHOLD
# -------------------------------------------------

def test_model_threshold():

    predictor = StudentRiskPredictor()

    assert predictor.threshold == 0.50

import shap
"""
Phase 9: Model Inference

Loads the trained model and predicts whether
a student may be at risk in the next assessment.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# -------------------------------------------------
# 1. PROJECT PATHS
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase6"
    / "final_model.joblib"
)


# -------------------------------------------------
# 2. FEATURE CONFIGURATION
# -------------------------------------------------

HISTORICAL_SCORE_COLUMNS = [
    "previous_score_1",
    "previous_score_2",
    "previous_score_3"
]

EXPECTED_FEATURES = [
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


# -------------------------------------------------
# 3. STUDENT FEATURE ENGINEERING
# -------------------------------------------------

def create_student_features(
    attendance_rate,
    assignment_completion_rate,
    previous_score_1,
    previous_score_2,
    previous_score_3
):

    scores = np.array(
        [
            previous_score_1,
            previous_score_2,
            previous_score_3
        ],
        dtype=float
    )

    # Validate input values
    if not np.isfinite(scores).all():
        raise ValueError("Assessment scores must be finite numbers.")

    if not np.isfinite([
        attendance_rate,
        assignment_completion_rate
    ]).all():
        raise ValueError("Attendance and assignment values must be finite.")

    if not 0 <= attendance_rate <= 100:
        raise ValueError("Attendance must be between 0 and 100.")

    if not 0 <= assignment_completion_rate <= 100:
        raise ValueError("Assignment completion must be between 0 and 100.")

    if not ((scores >= 0) & (scores <= 100)).all():
        raise ValueError("Assessment scores must be between 0 and 100.")

    # Same feature calculations as Phase 4
    average_previous_score = np.mean(scores)

    score_trend = (
        previous_score_1 - previous_score_3
    ) / 2

    recent_score_change = (
        previous_score_1 - previous_score_2
    )

    score_volatility = np.std(
        scores,
        ddof=0
    )

    student_features = {
        "attendance_rate": attendance_rate,
        "assignment_completion_rate": assignment_completion_rate,
        "previous_score_1": previous_score_1,
        "previous_score_2": previous_score_2,
        "previous_score_3": previous_score_3,
        "average_previous_score": average_previous_score,
        "score_trend": score_trend,
        "recent_score_change": recent_score_change,
        "score_volatility": score_volatility
    }

    return pd.DataFrame(
        [student_features],
        columns=EXPECTED_FEATURES
    )


# -------------------------------------------------
# 4. STUDENT RISK PREDICTOR
# -------------------------------------------------

class StudentRiskPredictor:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {MODEL_PATH}"
            )

        self.artifact = joblib.load(MODEL_PATH)

        self.model = self.artifact["model"]

        self.threshold = float(
            self.artifact["threshold"]
        )

        self.feature_columns = self.artifact["feature_columns"]

        if list(self.feature_columns) != EXPECTED_FEATURES:
            raise ValueError(
                "Model feature columns do not match "
                "the inference feature configuration."
            )

        print("Model loaded successfully.")
        print("Model:", self.artifact["model_name"])
        print("Prediction threshold:", self.threshold)


    # -------------------------------------------------
    # 5. GENERATE PREDICTION
    # -------------------------------------------------

    def predict(
        self,
        attendance_rate,
        assignment_completion_rate,
        previous_score_1,
        previous_score_2,
        previous_score_3
    ):

        X = create_student_features(
            attendance_rate,
            assignment_completion_rate,
            previous_score_1,
            previous_score_2,
            previous_score_3
        )

        probability = float(
            self.model.predict_proba(X)[0, 1]
        )

        prediction = int(
            probability >= self.threshold
        )

        # Advisory risk bands
        if probability >= 0.75:
            risk_band = "High"

        elif probability >= 0.50:
            risk_band = "Moderate"

        else:
            risk_band = "Low"

        return {
            "risk_probability": probability,
            "prediction": prediction,
            "risk_band": risk_band,
            "threshold": self.threshold,
            "features": X.iloc[0].to_dict()
        }

    # -------------------------------------------------
    # 6. GENERATE SHAP EXPLANATIONS
    # -------------------------------------------------

    def explain(self, features):

        # Extract the trained Random Forest estimator
        random_forest = self.model.named_steps["model"]

        # Create SHAP explainer
        explainer = shap.TreeExplainer(random_forest)

        # Calculate SHAP values
        shap_values = explainer.shap_values(features)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            risk_shap_values = shap_values[1][0]

        elif isinstance(shap_values, np.ndarray):

            if shap_values.ndim == 3:
                risk_shap_values = shap_values[0, :, 1]

            elif shap_values.ndim == 2:
                risk_shap_values = shap_values[0]

            else:
                raise ValueError(
                    f"Unexpected SHAP shape: {shap_values.shape}"
                )

        else:
            raise TypeError("Unsupported SHAP output format.")

        # Create explanation dataframe
        explanation_df = pd.DataFrame({
            "feature": self.feature_columns,
            "shap_value": risk_shap_values
        })

        explanation_df["direction"] = np.where(
            explanation_df["shap_value"] > 0,
            "Increases at-risk model output",
            "Decreases at-risk model output"
        )

        # Select five most influential features
        explanation_df["absolute_impact"] = (
            explanation_df["shap_value"].abs()
        )

        explanation_df = (
            explanation_df
            .sort_values("absolute_impact", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )

        return explanation_df


def generate_support_recommendations(features):

    recommendations = []

    if features["attendance_rate"] < 75:
        recommendations.append(
            "Attendance is below 75%. "
            "Discuss possible attendance barriers and "
            "available support with the student."
        )

    if features["assignment_completion_rate"] < 70:
        recommendations.append(
            "Assignment completion is below 70%. "
            "Consider assignment planning and time-management support."
        )

    if features["score_trend"] < -5:
        recommendations.append(
            "Previous assessment scores show a declining trend. "
            "Consider a supportive academic progress review."
        )

    if features["score_volatility"] > 15:
        recommendations.append(
            "Assessment scores show substantial variation. "
            "Review learning consistency and possible support needs."
        )

    if not recommendations:
        recommendations.append(
            "No predefined academic support trigger was identified. "
            "Continue routine monitoring and review the student's "
            "individual circumstances."
        )

    return recommendations

import pandas as pd

import streamlit as st

import pandas as pd

from src.inference.predictor import generate_support_recommendations

from src.inference.predictor import StudentRiskPredictor


# -------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Student Performance Risk Prediction",
    page_icon="🎓",
    layout="wide"
)


# -------------------------------------------------
# 2. LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_predictor():
    return StudentRiskPredictor()


predictor = load_predictor()


# -------------------------------------------------
# 3. DASHBOARD HEADER
# -------------------------------------------------

st.title("🎓 Student Performance & At-Risk Prediction")

st.write(
    "An AI-powered advisory system to identify students "
    "who may benefit from additional academic support."
)

st.divider()


# -------------------------------------------------
# 4. STUDENT INPUT FORM
# -------------------------------------------------

st.subheader("Enter Student Performance Information")

col1, col2 = st.columns(2)

with col1:

    attendance = st.number_input(
        "Attendance Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

    assignment = st.number_input(
        "Assignment Completion (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

with col2:

    score_1 = st.number_input(
        "Previous Assessment 1",
        min_value=0.0,
        max_value=100.0,
        value=45.0
    )

    score_2 = st.number_input(
        "Previous Assessment 2",
        min_value=0.0,
        max_value=100.0,
        value=48.0
    )

    score_3 = st.number_input(
        "Previous Assessment 3",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )


# -------------------------------------------------
# 5. GENERATE PREDICTION
# -------------------------------------------------

if st.button("Predict Student Risk", type="primary"):

    try:

        result = predictor.predict(
            attendance_rate=attendance,
            assignment_completion_rate=assignment,
            previous_score_1=score_1,
            previous_score_2=score_2,
            previous_score_3=score_3
        )
        # Generate SHAP explanations for this student
        explanation_df = predictor.explain(
          pd.DataFrame([result["features"]])
        )

        probability = result["risk_probability"] * 100

        st.divider()

        st.subheader("Prediction Results")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Risk Probability",
            f"{probability:.2f}%"
        )

        metric2.metric(
            "Prediction",
            "At Risk" if result["prediction"] == 1
            else "Not At Risk"
        )

        metric3.metric(
            "Advisory Risk Band",
            result["risk_band"]
        )

        # Risk interpretation
        if result["risk_band"] == "High":

            st.error(
                "High advisory risk: Consider a supportive "
                "academic review."
            )

        elif result["risk_band"] == "Moderate":

            st.warning(
                "Moderate advisory risk: Consider checking "
                "the student's recent learning progress."
            )

        else:

            st.success(
                "Low advisory risk: Continue routine "
                "academic monitoring."
            )

        # Feature information
        st.subheader("Calculated Model Features")

        feature_df = result["features"]

        st.dataframe(
            {
                "Feature": list(feature_df.keys()),
                "Value": [
                    round(float(value), 3)
                    for value in feature_df.values()
                ]
            },
            width="stretch",
            hide_index=True
        )

        # -------------------------------------------------
        # SHAP EXPLANATIONS
        # -------------------------------------------------

        st.divider()

        st.subheader("Why Did the Model Make This Prediction?")

        st.write(
            "The following features had the greatest influence "
            "on the model's prediction for this student."
        )

        st.caption(
            "Positive SHAP values push the model output toward "
            "the at-risk class. Negative values push it away."
        )

        # Display top five features
        st.dataframe(
            explanation_df[
                ["feature", "shap_value", "direction"]
            ].rename(columns={
                "feature": "Feature",
                "shap_value": "SHAP Contribution",
                "direction": "Influence"
            }),
            width="stretch",
            hide_index=True
        )

        # Display SHAP bar chart
        chart_df = (
            explanation_df
            .set_index("feature")[["shap_value"]]
            .sort_values("shap_value")
        )

        st.subheader("Feature Contribution Chart")

        st.bar_chart(
            chart_df,
            x_label="SHAP contribution",
            horizontal=True
        )

        st.caption(
            "SHAP contributions explain model output, not "
            "causation or percentage-point changes in risk."
        )

        # -------------------------------------------------
        # SUPPORT RECOMMENDATIONS
        # -------------------------------------------------
        # -------------------------------------------------
        # -------------------------------------------------

        st.divider()

        st.subheader("Suggested Academic Support")

        recommendations = generate_support_recommendations(
          result["features"]
        )

        for recommendation in recommendations:
          st.info(recommendation)

        # Responsible AI notice
        st.divider()

        st.subheader("Human Review Required")

        st.info(
            "This model provides advisory predictions only. "
            "A qualified educator must review the student's "
            "individual circumstances before taking any action. "
            "Never use this prediction for punishment, grading, "
            "admission, or other automated high-stakes decisions."
        )

    except (ValueError, TypeError) as error:

        st.error(f"Invalid student input: {error}")


# -------------------------------------------------
# 6. FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Student Performance & At-Risk Prediction | "
    "AI/ML Decision Support System"
)


from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Saved model
MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase6"
    / "final_model.joblib"
)

# Output directory
OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "phase8"
)


def load_model():

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    print("Model loaded successfully.")
    print("Model:", artifact["model_name"])

    return model, feature_columns


def calculate_feature_importance(model, feature_columns):

    """
    Extract Random Forest feature importance.
    """

    # Access Random Forest inside Pipeline
    random_forest = model.named_steps["model"]

    importance_values = random_forest.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance_values
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========== FEATURE IMPORTANCE ==========")
    print(importance_df.to_string(index=False))

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    importance_df.to_csv(
        OUTPUT_PATH / "feature_importance.csv",
        index=False
    )

    return importance_df


def plot_feature_importance(importance_df):

    """
    Create and save a horizontal feature importance chart.
    """

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    plot_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        plot_df["Feature"],
        plot_df["Importance"]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Random Forest Feature Importance")

    plt.tight_layout()

    chart_path = OUTPUT_PATH / "feature_importance.png"

    plt.savefig(chart_path, dpi=300, bbox_inches="tight")

    plt.close()

    print("\nFeature importance chart saved:")
    print(chart_path)


def calculate_shap_values(model, feature_columns):

    """
    Calculate SHAP values for the saved Random Forest.
    """

    test_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "splits"
        / "test.csv"
    )

    test_df = pd.read_csv(test_path)

    X_test = test_df[feature_columns].copy()

    # Explain a sample to reduce computation time
    X_sample = X_test.sample(
        n=min(200, len(X_test)),
        random_state=42
    )

    # Access Random Forest from pipeline
    random_forest = model.named_steps["model"]

    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(random_forest)

    shap_values = explainer.shap_values(X_sample)

    print("SHAP calculation completed.")

    print("Sample shape:", X_sample.shape)

    return explainer, shap_values, X_sample


def plot_shap_summary(shap_values, X_sample, feature_columns):

    """
    Generate global SHAP summary visualizations.
    """

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Select class 1 (At Risk)
    risk_shap_values = shap_values[:, :, 1]

    # Beeswarm summary plot
    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        risk_shap_values,
        X_sample,
        feature_names=feature_columns,
        show=False
    )

    plt.title("SHAP Summary Plot - At Risk Class")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH / "shap_summary_beeswarm.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Mean absolute SHAP importance
    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        risk_shap_values,
        X_sample,
        feature_names=feature_columns,
        plot_type="bar",
        show=False
    )

    plt.title("SHAP Feature Importance - At Risk Class")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH / "shap_summary_bar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSHAP summary plots saved successfully.")
    print("Beeswarm:", OUTPUT_PATH / "shap_summary_beeswarm.png")
    print("Bar plot:", OUTPUT_PATH / "shap_summary_bar.png")


def explain_individual_students(
    model,
    explainer,
    shap_values,
    X_sample,
    feature_columns,
    top_n=5
):
    """
    Generate individual SHAP explanations
    for students predicted as at risk.
    """

    risk_shap_values = shap_values[:, :, 1]

    probabilities = model.predict_proba(X_sample)[:, 1]

    threshold = 0.50

    predictions = (probabilities >= threshold).astype(int)

    explanation_records = []

    for i in range(len(X_sample)):

        if predictions[i] == 1:

            student_values = risk_shap_values[i]

            importance_order = np.argsort(
                np.abs(student_values)
            )[::-1][:top_n]

            for rank, feature_index in enumerate(
                importance_order, start=1
            ):

                explanation_records.append({
                    "sample_index": i,
                    "risk_probability": probabilities[i],
                    "predicted_at_risk": predictions[i],
                    "rank": rank,
                    "feature": feature_columns[feature_index],
                    "feature_value": X_sample.iloc[
                        i, feature_index
                    ],
                    "shap_value": student_values[feature_index],
                    "direction": (
                        "Increases model output for at-risk class"
                        if student_values[feature_index] > 0
                        else "Decreases model output for at-risk class"
                    )
                })

    explanation_df = pd.DataFrame(explanation_records)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    explanation_df.to_csv(
        OUTPUT_PATH / "individual_student_explanations.csv",
        index=False
    )

    print("\nIndividual student explanations generated.")
    print("Students explained:", explanation_df["sample_index"].nunique())

    print("\nSample explanations:")
    print(explanation_df.head(15).to_string(index=False))

    return explanation_df


def explain_all_flagged_students(model, feature_columns):

    """
    Generate SHAP explanations for every student
    flagged as at risk in the Period 5 test dataset.
    """

    test_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "splits"
        / "test.csv"
    )

    test_df = pd.read_csv(test_path)

    X_test = test_df[feature_columns].copy()

    threshold = 0.50

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    flagged_positions = np.where(predictions == 1)[0]

    X_flagged = X_test.iloc[flagged_positions].copy()

    print("\nTotal test students:", len(X_test))
    print("Students flagged as at risk:", len(X_flagged))

    if len(X_flagged) == 0:
        print("No students flagged as at risk.")
        return pd.DataFrame()

    # Explain only the flagged students
    random_forest = model.named_steps["model"]

    explainer = shap.TreeExplainer(random_forest)

    flagged_shap = explainer.shap_values(X_flagged)

    # Select at-risk class (class 1)
    risk_shap = flagged_shap[:, :, 1]

    records = []

    for i, original_position in enumerate(flagged_positions):

        student_id = test_df.iloc[original_position]["student_id"]

        top_features = np.argsort(
            np.abs(risk_shap[i])
        )[::-1][:5]

        for rank, feature_index in enumerate(top_features, start=1):

            records.append({
                "student_id": student_id,
                "risk_probability": probabilities[original_position],
                "predicted_at_risk": 1,
                "feature_rank": rank,
                "feature": feature_columns[feature_index],
                "feature_value": X_flagged.iloc[i, feature_index],
                "shap_value": risk_shap[i, feature_index],
                "direction": (
                    "Pushes toward at-risk"
                    if risk_shap[i, feature_index] > 0
                    else "Pushes away from at-risk"
                )
            })

    explanations_df = pd.DataFrame(records)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    explanations_df.to_csv(
        OUTPUT_PATH / "all_flagged_student_explanations.csv",
        index=False
    )

    print("\nAll flagged student explanations saved.")
    print("Unique students explained:",
          explanations_df["student_id"].nunique())

    print("\nFirst 10 explanation records:")
    print(explanations_df.head(10).to_string(index=False))

    return explanations_df


def main():

    print("PHASE 8: MODEL EXPLAINABILITY")

    model, feature_columns = load_model()

    importance_df = calculate_feature_importance(
        model,
        feature_columns
    )
    plot_feature_importance(importance_df)

    explainer, shap_values, X_sample = calculate_shap_values(
    model,
    feature_columns
    )

    plot_shap_summary(
    shap_values,
    X_sample,
    feature_columns
    )

    explanation_df = explain_individual_students(
    model,
    explainer,
    shap_values,
    X_sample,
    feature_columns,
    top_n=5
    )

    all_flagged_explanations = explain_all_flagged_students(
    model,
    feature_columns
    )

    print("\nSHAP output type:", type(shap_values))

    if isinstance(shap_values, list):
        print("SHAP output shape:", [
            values.shape for values in shap_values
    ])
    else:
        print("SHAP output shape:", shap_values.shape)

    print("\nPhase 8.1 completed successfully.")


if __name__ == "__main__":
    main()

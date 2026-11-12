
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "phase8"
    / "all_flagged_student_explanations.csv"
)

OUTPUT_PATH = PROJECT_ROOT / "reports" / "phase8"


def assign_risk_band(probability):

    if probability >= 0.75:
        return "High"

    elif probability >= 0.50:
        return "Moderate"

    return "Low"


def generate_advisory_report():

    df = pd.read_csv(INPUT_PATH)

    # One record per student for the advisory summary
    student_report = (
        df.groupby("student_id")
        .agg(
            risk_probability=("risk_probability", "first"),
            predicted_at_risk=("predicted_at_risk", "first")
        )
        .reset_index()
    )

    student_report["risk_band"] = (
        student_report["risk_probability"]
        .apply(assign_risk_band)
    )

    # Summarize the leading SHAP contributors
    explanations = []

    for student_id, group in df.groupby("student_id"):

        top_factors = group.sort_values(
            "feature_rank"
        )["feature"].head(3).tolist()

        explanations.append({
            "student_id": student_id,
            "top_contributing_factors": ", ".join(top_factors)
        })

    explanations_df = pd.DataFrame(explanations)

    student_report = student_report.merge(
        explanations_df,
        on="student_id",
        how="left"
    )

    student_report["recommended_action"] = (
        "Offer supportive academic check-in and review "
        "recent learning needs."
    )

    student_report["human_review_required"] = True

    student_report["disclaimer"] = (
        "Advisory prediction only. Not for grading, "
        "punishment, admission, or other automated decisions."
    )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_PATH / "responsible_ai_advisory_report.csv"

    student_report.to_csv(output_file, index=False)

    print("\n========== RESPONSIBLE AI REPORT ==========")

    print("Students included:", len(student_report))

    print("\nRisk band distribution:")
    print(student_report["risk_band"].value_counts())

    print("\nSample advisory records:")
    print(student_report.head(10).to_string(index=False))

    print("\nReport saved at:")
    print(output_file)


if __name__ == "__main__":
    generate_advisory_report()

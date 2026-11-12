
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths
MODEL_PATH = PROJECT_ROOT / "artifacts" / "phase6" / "final_model.joblib"

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "test.csv"
)

def load_model():
    """Load the saved Phase 6 model artifact."""

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_columns = artifact["feature_columns"]

    print("Model loaded successfully.")
    print("Model name:", artifact["model_name"])
    print("Operational threshold:", threshold)
    print("Number of features:", len(feature_columns))

    return model, threshold, feature_columns


def load_test_data():
    """Load the untouched Period 5 test dataset."""

    test_df = pd.read_csv(TEST_PATH)

    print("\nTest dataset loaded successfully.")
    print("Test dataset shape:", test_df.shape)

    print("\nPrediction period distribution:")
    print(test_df["prediction_period"].value_counts().sort_index())

    return test_df


def prepare_test_data(test_df, feature_columns):
    """
    Prepare Period 5 features and actual target labels.
    """

    target_column = "at_risk_next_assessment"

    # Check required columns
    required_columns = feature_columns + [target_column]

    missing_columns = [
        col for col in required_columns
        if col not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Separate input features and actual labels
    X_test = test_df[feature_columns].copy()
    y_test = test_df[target_column].copy()

    # Check missing values
    print("\nMissing values in test features:")
    print(X_test.isnull().sum())

    if X_test.isnull().any().any():
        raise ValueError("Missing values detected in test features.")

    if y_test.isnull().any():
        raise ValueError("Missing values detected in target.")

    # Check target classes
    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    print("\nTest feature preparation completed.")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nActual target distribution:")
    print(y_test.value_counts().sort_index())

    return X_test, y_test


def generate_predictions(model, threshold, X_test):

    """
    Generate final predictions using the saved model
    and selected operational threshold.
    """

    # Get probability of belonging to at-risk class
    probabilities = model.predict_proba(X_test)[:, 1]

    # Apply the saved classification threshold
    predictions = (probabilities >= threshold).astype(int)

    print("\nFinal predictions generated successfully.")

    print("\nPrediction distribution:")
    print(pd.Series(predictions).value_counts().sort_index())

    print("\nFirst 10 predicted probabilities:")
    print(np.round(probabilities[:10], 4))

    return predictions, probabilities


def evaluate_model(y_test, predictions, probabilities):

    """
    Calculate final model performance metrics.
    """

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    average_precision = average_precision_score(
        y_test, probabilities
    )

    cm = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0

    print("\n========== FINAL TEST RESULTS ==========")

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"Average Precision: {average_precision:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Not at Risk", "At Risk"],
            zero_division=0
        )
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "Average_Precision": average_precision,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }



def save_final_reports(test_df, predictions, probabilities, metrics):

    output_path = PROJECT_ROOT / "reports" / "phase7"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save evaluation metrics
    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        output_path / "final_test_metrics.csv",
        index=False
    )

    # Save individual student predictions
    predictions_df = pd.DataFrame({
        "student_id": test_df["student_id"],
        "actual_at_risk": test_df["at_risk_next_assessment"],
        "predicted_at_risk": predictions,
        "predicted_probability": probabilities
    })

    predictions_df.to_csv(
        output_path / "final_test_predictions.csv",
        index=False
    )

    # Save confusion matrix
    cm = confusion_matrix(
        test_df["at_risk_next_assessment"],
        predictions,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Not_At_Risk", "Actual_At_Risk"],
        columns=["Predicted_Not_At_Risk", "Predicted_At_Risk"]
    )

    cm_df.to_csv(
        output_path / "confusion_matrix.csv"
    )

    print("\nFinal evaluation reports saved successfully.")
    print("Reports directory:", output_path)



def main():

    print("PHASE 7: FINAL MODEL TESTING")

    model, threshold, feature_columns = load_model()

    test_df = load_test_data()

    X_test, y_test = prepare_test_data(
        test_df,
        feature_columns
    )

    predictions, probabilities = generate_predictions(
        model,
        threshold,
        X_test
    )

    metrics = evaluate_model(y_test, predictions, probabilities)

    save_final_reports(test_df, predictions, probabilities, metrics)

    print("\nPhase 7 prediction generation completed.")


if __name__ == "__main__":
    main()
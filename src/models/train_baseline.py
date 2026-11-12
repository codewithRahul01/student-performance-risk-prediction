
"""
PHASE 5: BASELINE MODEL DEVELOPMENT

Project:
Student Performance & At-Risk Prediction

Models:
1. Dummy Classifier (baseline)
2. Logistic Regression
3. Random Forest

Training:
Prediction periods 1, 2, 3

Validation:
Prediction period 4

Test:
Prediction period 5 (NOT USED in this script)

Evaluation:
- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- Average Precision
- Confusion Matrix

Random seed: 42
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "phase5"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase5"
)

RANDOM_SEED = 42

TARGET_COLUMN = "at_risk_next_assessment"


# =====================================================
# 2. MODEL INPUT FEATURES
# =====================================================

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


# =====================================================
# 3. LOAD TRAINING AND VALIDATION DATA
# =====================================================

def load_data():

    train_path = SPLIT_DIR / "train.csv"

    validation_path = SPLIT_DIR / "validation.csv"

    if not train_path.exists():

        raise FileNotFoundError(
            "Training dataset not found. "
            "Run create_splits.py first."
        )

    if not validation_path.exists():

        raise FileNotFoundError(
            "Validation dataset not found. "
            "Run create_splits.py first."
        )

    train_df = pd.read_csv(train_path)

    validation_df = pd.read_csv(validation_path)

    print("\nDatasets loaded successfully.")

    print("Training shape:", train_df.shape)

    print("Validation shape:", validation_df.shape)

    return train_df, validation_df


# =====================================================
# 4. PREPARE X AND y
# =====================================================

def prepare_data(train_df, validation_df):

    required_columns = (
        MODEL_FEATURES + [TARGET_COLUMN]
    )

    for column in required_columns:

        if column not in train_df.columns:

            raise ValueError(
                f"Missing training column: {column}"
            )

        if column not in validation_df.columns:

            raise ValueError(
                f"Missing validation column: {column}"
            )

    X_train = train_df[MODEL_FEATURES].copy()

    y_train = train_df[TARGET_COLUMN].astype(int)

    X_validation = (
        validation_df[MODEL_FEATURES].copy()
    )

    y_validation = (
        validation_df[TARGET_COLUMN].astype(int)
    )

    # Check for missing values.
    if X_train.isnull().any().any():

        raise ValueError(
            "Missing values found in training features."
        )

    if X_validation.isnull().any().any():

        raise ValueError(
            "Missing values found in validation features."
        )

    # Check target classes.
    if set(y_train.unique()) != {0, 1}:

        raise ValueError(
            "Training data must contain both classes."
        )

    if not set(y_validation.unique()).issubset({0, 1}):

        raise ValueError(
            "Unexpected validation target values."
        )

    # Prevent accidental future leakage.
    forbidden = {
        "next_assessment_score",
        TARGET_COLUMN,
        "student_id"
    }

    if forbidden.intersection(MODEL_FEATURES):

        raise ValueError(
            "Forbidden feature detected."
        )

    print("\nFeature preparation completed.")

    print("Number of input features:", len(MODEL_FEATURES))

    print("\nTraining class distribution:")

    print(y_train.value_counts())

    print("\nValidation class distribution:")

    print(y_validation.value_counts())

    return (
        X_train,
        y_train,
        X_validation,
        y_validation
    )


# =====================================================
# 5. DEFINE MODELS
# =====================================================

def build_models():

    models = {

        # ---------------------------------------------
        # MODEL 1: DUMMY CLASSIFIER
        # ---------------------------------------------

        "Dummy_Baseline": Pipeline([

            (
                "classifier",
                DummyClassifier(
                    strategy="most_frequent"
                )
            )

        ]),

        # ---------------------------------------------
        # MODEL 2: LOGISTIC REGRESSION
        # ---------------------------------------------

        "Logistic_Regression": Pipeline([

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_SEED
                )
            )

        ]),

        # ---------------------------------------------
        # MODEL 3: RANDOM FOREST
        # ---------------------------------------------

        "Random_Forest": Pipeline([

            (
                "classifier",

                RandomForestClassifier(

                    n_estimators=400,

                    max_depth=None,

                    min_samples_split=4,

                    min_samples_leaf=2,

                    class_weight="balanced_subsample",

                    random_state=RANDOM_SEED,

                    n_jobs=-1

                )

            )

        ])

    }

    return models


# =====================================================
# 6. EVALUATE MODEL
# =====================================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
    model_name
):

    # Predict class labels at default threshold.
    predictions = model.predict(X_validation)

    # Predict probabilities for positive class.
    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # Confusion matrix.
    cm = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # Classification metrics.
    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    avg_precision = average_precision_score(
        y_validation,
        probabilities
    )

    results = {

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "Specificity": specificity,

        "Average_Precision": avg_precision,

        "True_Negatives": tn,

        "False_Positives": fp,

        "False_Negatives": fn,

        "True_Positives": tp

    }

    return results, cm, predictions, probabilities


# =====================================================
# 7. PLOT CONFUSION MATRIX
# =====================================================

def save_confusion_matrix(cm, model_name):

    plt.figure(figsize=(7, 5))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=[
            "Not at Risk",
            "At Risk"
        ],

        yticklabels=[
            "Not at Risk",
            "At Risk"
        ]

    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel("Predicted Label")

    plt.ylabel("Actual Label")

    plt.tight_layout()

    filename = (
        model_name.lower()
        + "_confusion_matrix.png"
    )

    plt.savefig(
        REPORT_DIR / filename,
        dpi=300
    )

    plt.close()


# =====================================================
# 8. TRAIN AND EVALUATE ALL MODELS
# =====================================================

def train_models(
    X_train,
    y_train,
    X_validation,
    y_validation
):

    models = build_models()

    all_results = []

    validation_predictions = pd.DataFrame({

        "actual_target": y_validation.to_numpy()

    })

    for model_name, model in models.items():

        print("\n" + "=" * 55)

        print("Training:", model_name)

        print("=" * 55)

        # Fit only on training data.
        model.fit(
            X_train,
            y_train
        )

        print("Training completed.")

        # Evaluate on validation data.
        (
            results,
            cm,
            predictions,
            probabilities
        ) = evaluate_model(

            model,

            X_validation,

            y_validation,

            model_name

        )

        all_results.append(results)

        validation_predictions[
            f"{model_name}_prediction"
        ] = predictions

        validation_predictions[
            f"{model_name}_probability"
        ] = probabilities

        # Save model artifact.
        model_path = (
            ARTIFACT_DIR
            / f"{model_name.lower()}.joblib"
        )

        joblib.dump(
            model,
            model_path
        )

        # Save confusion matrix.
        save_confusion_matrix(
            cm,
            model_name
        )

        print("\nValidation Results:")

        for metric, value in results.items():

            print(f"{metric}: {value}")

    return all_results, validation_predictions


# =====================================================
# 9. SAVE EXPERIMENT RESULTS
# =====================================================

def save_results(
    all_results,
    validation_predictions
):

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="F1_Score",
        ascending=False
    )

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    validation_predictions.to_csv(
        REPORT_DIR / "validation_predictions.csv",
        index=False
    )

    print("\nModel comparison:")

    print(
        results_df.to_string(index=False)
    )

    print("\nResults saved successfully.")

    print(
        "Comparison:",
        REPORT_DIR / "model_comparison.csv"
    )

    print(
        "Predictions:",
        REPORT_DIR / "validation_predictions.csv"
    )

    return results_df


# =====================================================
# 10. MAIN EXECUTION
# =====================================================

def main():

    print("\nPHASE 5: MODEL DEVELOPMENT")

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df, validation_df = load_data()

    (
        X_train,
        y_train,
        X_validation,
        y_validation
    ) = prepare_data(
        train_df,
        validation_df
    )

    all_results, validation_predictions = train_models(

        X_train,

        y_train,

        X_validation,

        y_validation

    )

    results_df = save_results(

        all_results,

        validation_predictions

    )

    print("\nPHASE 5 COMPLETED!")

    print("\nBest validation F1-score:")

    print(
        results_df.iloc[0]["Model"],
        results_df.iloc[0]["F1_Score"]
    )

    print(
        "\nNOTE: The test dataset was not used."
    )


if __name__ == "__main__":

    main()

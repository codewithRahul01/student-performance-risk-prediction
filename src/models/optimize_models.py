
"""
PHASE 6: MODEL OPTIMIZATION

Project:
Student Performance & At-Risk Prediction

Objectives:
1. Tune Logistic Regression.
2. Tune Random Forest.
3. Use chronological cross-validation.
4. Compare models on validation period 4.
5. Optimize prediction thresholds.
6. Keep period 5 untouched.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    make_scorer,
    f1_score,
    average_precision_score
)


FEATURE_COLUMNS = [
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
# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

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
    / "phase6"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase6"
)

RANDOM_SEED = 42

TARGET_COLUMN = "at_risk_next_assessment"


# ==================================================
# 2. MODEL FEATURES
# ==================================================

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


# ==================================================
# 3. LOAD TRAINING AND VALIDATION DATA
# ==================================================

def load_data():

    train_path = SPLIT_DIR / "train.csv"

    validation_path = (
        SPLIT_DIR / "validation.csv"
    )

    if not train_path.exists():

        raise FileNotFoundError(
            "Training dataset not found."
        )

    if not validation_path.exists():

        raise FileNotFoundError(
            "Validation dataset not found."
        )

    train_df = pd.read_csv(train_path)

    validation_df = pd.read_csv(
        validation_path
    )

    print("\nDatasets loaded successfully.")

    print("Training:", train_df.shape)

    print("Validation:", validation_df.shape)

    return train_df, validation_df


# ==================================================
# 4. INSPECT TEMPORAL CLASS DISTRIBUTION
# ==================================================

def inspect_periods(train_df):

    distribution = pd.crosstab(
        train_df["prediction_period"],
        train_df[TARGET_COLUMN]
    ).reindex(
        columns=[0, 1],
        fill_value=0
    )

    print("\nClass distribution by period:")

    print(distribution)

    for period in [1, 2, 3]:

        counts = distribution.loc[period]

        if (counts == 0).any():

            raise ValueError(
                f"Period {period} does not contain "
                "both classes. Reconsider temporal "
                "cross-validation."
            )

    print(
        "\nBoth classes are present in each "
        "training period."
    )


# ==================================================
# 5. PREPARE FEATURES
# ==================================================

def prepare_data(train_df, validation_df):

    X_train = train_df[
        MODEL_FEATURES
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].astype(int)

    X_validation = validation_df[
        MODEL_FEATURES
    ].copy()

    y_validation = validation_df[
        TARGET_COLUMN
    ].astype(int)

    if X_train.isnull().any().any():

        raise ValueError(
            "Missing training feature values."
        )

    if X_validation.isnull().any().any():

        raise ValueError(
            "Missing validation feature values."
        )

    if not np.isfinite(
        X_train.to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Infinite values in training features."
        )

    if not np.isfinite(
        X_validation.to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Infinite values in validation features."
        )

    print("\nFeature preparation completed.")

    print("Training features:", X_train.shape)

    print("Validation features:", X_validation.shape)

    return (
        X_train,
        y_train,
        X_validation,
        y_validation
    )


# ==================================================
# 6. CREATE CHRONOLOGICAL CV FOLDS
# ==================================================

def create_temporal_cv(train_df):

    periods = train_df[
        "prediction_period"
    ].to_numpy()

    cv_splits = []

    # Fold 1: Train period 1, validate period 2
    # Fold 2: Train periods 1 and 2,
    #         validate period 3

    for validation_period in [2, 3]:

        train_indices = np.where(
            periods < validation_period
        )[0]

        validation_indices = np.where(
            periods == validation_period
        )[0]

        cv_splits.append(
            (
                train_indices,
                validation_indices
            )
        )

    print("\nTemporal CV folds created:")

    for i, (train_idx, val_idx) in enumerate(
        cv_splits,
        start=1
    ):

        print(
            f"Fold {i}: "
            f"Train={len(train_idx)}, "
            f"Validation={len(val_idx)}"
        )

    return cv_splits



def tune_logistic_regression(X_train, y_train, temporal_cv):
    """
    Tune Logistic Regression using expanding-window
    temporal cross-validation.
    """

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        ))
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__solver": ["liblinear", "lbfgs"]
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=temporal_cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    print("\nStarting Logistic Regression tuning...")

    search.fit(X_train, y_train)

    print("\nLogistic Regression tuning completed.")
    print("Best parameters:", search.best_params_)
    print("Best CV Average Precision:", round(search.best_score_, 4))

    return search



def tune_random_forest(X_train, y_train, temporal_cv):
    """
    Tune Random Forest using expanding-window
    temporal cross-validation.
    """

    pipeline = Pipeline([
        ("model", RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 0.8]
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=temporal_cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    print("\nStarting Random Forest tuning...")

    search.fit(X_train, y_train)

    print("\nRandom Forest tuning completed.")
    print("Best parameters:", search.best_params_)
    print("Best CV Average Precision:", round(search.best_score_, 4))

    return search


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


def evaluate_tuned_models(
    logistic_search,
    random_forest_search,
    X_validation,
    y_validation
):
    """
    Evaluate the best tuned models on validation Period 4.
    """

    models = {
        "Tuned Logistic Regression": logistic_search.best_estimator_,
        "Tuned Random Forest": random_forest_search.best_estimator_
    }

    results = []

    for model_name, model in models.items():

        predictions = model.predict(X_validation)
        probabilities = model.predict_proba(X_validation)[:, 1]

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_validation, predictions),
            "Precision": precision_score(
                y_validation, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_validation, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_validation, predictions, zero_division=0
            ),
            "Average_Precision": average_precision_score(
                y_validation, probabilities
            )
        }

        results.append(metrics)

        print(f"\n{model_name}")
        for metric, value in metrics.items():
            if metric != "Model":
                print(f"{metric}: {value:.4f}")

        print("Confusion Matrix:")
        print(confusion_matrix(y_validation, predictions))

    results_df = pd.DataFrame(results)

    output_path = PROJECT_ROOT / "reports" / "phase6"
    output_path.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        output_path / "tuned_model_comparison.csv",
        index=False
    )

    print("\nTuned model comparison saved.")

    return results_df


def optimize_threshold(random_forest_search, X_validation, y_validation):

    model = random_forest_search.best_estimator_

    probabilities = model.predict_proba(X_validation)[:, 1]

    threshold_results = []

    for threshold in np.arange(0.10, 0.91, 0.05):

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_validation, predictions, zero_division=0
        )

        recall = recall_score(
            y_validation, predictions, zero_division=0
        )

        f1 = f1_score(
            y_validation, predictions, zero_division=0
        )

        threshold_results.append({
            "Threshold": round(threshold, 2),
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

    threshold_df = pd.DataFrame(threshold_results)

    print("\nThreshold comparison:")
    print(threshold_df.round(4).to_string(index=False))

    # Select the threshold with the highest F1 score
    best_row = threshold_df.loc[threshold_df["F1"].idxmax()]

    best_threshold = float(best_row["Threshold"])

    print("\nBest threshold:", best_threshold)
    print("Best Precision:", round(best_row["Precision"], 4))
    print("Best Recall:", round(best_row["Recall"], 4))
    print("Best F1:", round(best_row["F1"], 4))

    output_path = PROJECT_ROOT / "reports" / "phase6"
    output_path.mkdir(parents=True, exist_ok=True)

    threshold_df.to_csv(
        output_path / "threshold_comparison.csv",
        index=False
    )

    return best_threshold, threshold_df


def save_final_model(random_forest_search, best_threshold):

    output_path = PROJECT_ROOT / "artifacts" / "phase6"
    output_path.mkdir(parents=True, exist_ok=True)

    final_model = random_forest_search.best_estimator_

    artifact = {
        "model": final_model,
        "threshold": 0.50,
        "f1_optimal_threshold": best_threshold,
        "model_name": "Tuned Random Forest",
        "feature_columns": FEATURE_COLUMNS
    }

    model_path = output_path / "final_model.joblib"

    joblib.dump(artifact, model_path)

    print("\nFinal model saved successfully.")
    print("Model path:", model_path)
    print("Selected operational threshold: 0.50")
    print("F1-optimal threshold:", best_threshold)
    

# ==================================================
# 7. MAIN CONFIGURATION CHECK
# ==================================================

def main():

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df, validation_df = load_data()

    inspect_periods(train_df)

    (
        X_train,
        y_train,
        X_validation,
        y_validation
    ) = prepare_data(
        train_df,
        validation_df
    )

    cv_splits = create_temporal_cv(
        train_df
    )

    logistic_search = tune_logistic_regression(
    X_train,
    y_train,
    cv_splits
    )
    random_forest_search = tune_random_forest(
    X_train,
    y_train,
    cv_splits
    )
    results_df = evaluate_tuned_models(
    logistic_search,
    random_forest_search,
    X_validation,
    y_validation
  )
    

    print("\nFinal validation comparison:")
    print(results_df.to_string(index=False))

    best_threshold, threshold_df = optimize_threshold(
    random_forest_search,
    X_validation,
    y_validation
    )
    save_final_model(
    random_forest_search,
    best_threshold
    )
    print("\nModel optimization completed successfully.")


if __name__ == "__main__":

    main()


# Student Performance & At-Risk Prediction

An AI/ML-based academic decision-support system that identifies students who may benefit from additional academic support based on historical academic performance.

The system predicts whether a student may score below 40 in their next assessment using attendance, assignment completion and previous assessment scores.

**Important:** This is an advisory decision-support system. Predictions must be reviewed by educators and must never be used for automated grading, admissions, punishment or disciplinary decisions.

---

## 1. Project Overview

### Problem Statement

Educational institutions need ways to identify students who may benefit from additional academic support before their next assessment.

This project develops a machine learning classification system that predicts the likelihood of a student entering the at-risk category in their next assessment.

### Objectives

- Generate reproducible synthetic academic data.
- Perform exploratory data analysis and preprocessing.
- Engineer historical academic performance features.
- Compare multiple classification models.
- Optimize model hyperparameters and prediction thresholds.
- Evaluate performance on an untouched chronological test set.
- Explain predictions using SHAP.
- Provide an interactive Streamlit dashboard.
- Generate supportive academic recommendations.
- Include automated tests and responsible AI safeguards.

## 2. Dataset

The project uses reproducibly generated synthetic academic records.

| Dataset characteristic | Value |
|---|---:|
| Total records | 5,000 |
| Unique students | 1,000 |
| Prediction periods | 5 |
| At-risk records | 237 |
| Not-at-risk records | 4,763 |

### Target Definition

The target variable is `at_risk_next_assessment`.

- `1`: Next assessment score below 40.
- `0`: Next assessment score greater than or equal to 40.

### Input Features

- Attendance rate
- Assignment completion rate
- Previous assessment scores
- Average previous score
- Score trend
- Recent score change
- Score volatility

Student identifiers, dates and future assessment outcomes are excluded from model inputs.

See `data/DATA_DICTIONARY.md` for additional dataset information.

## 3. Technology Stack

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP
- Joblib
- Streamlit
- Pytest

## 4. Project Structure

```text
student-performance-risk-prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data/
│   │   ├── generate_data.py
│   │   ├── preprocess.py
│   │   └── create_splits.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train_baseline.py
│   │   ├── optimize_models.py
│   │   ├── evaluate_final.py
│   │   ├── explain_model.py
│   │   └── responsible_ai.py
│   │
│   └── inference/
│       ├── __init__.py
│       └── predictor.py
│
├── artifacts/
│
├── reports/
│   ├── figures/
│   ├── phase5/
│   ├── phase6/
│   ├── phase7/
│   ├── phase8/
│   └── pytest-results.xml
│
└── tests/
    ├── test_features.py
    ├── test_inference.py
    └── test_reproducibility.py
```

## 5. Exploratory Data Analysis

Exploratory analysis examines:

- Attendance distribution
- Assignment completion distribution
- Previous and next assessment scores
- Class distribution
- Attendance versus assessment performance
- Assignment completion versus assessment performance
- Correlation between numerical features
- Outlier analysis
- Performance trends across prediction periods

Visualizations are available in `reports/figures/`.

## 6. Feature Engineering

Four additional features are created from historical assessment scores.

| Feature | Calculation |
|---|---|
| Average previous score | Mean of three previous scores |
| Score trend | (Score 1 − Score 3) / 2 |
| Recent score change | Score 1 − Score 2 |
| Score volatility | Population standard deviation of three previous scores |

The implementation uses `ddof=0` for score volatility.

All predictive features use information available before the next assessment.

## 7. Model Development

Three initial classification approaches were compared:

1. Dummy Classifier
2. Logistic Regression
3. Random Forest

### Initial Validation Results

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Dummy Classifier | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.289 | 0.824 | 0.427 |
| Random Forest | 0.588 | 0.588 | 0.588 |

Random Forest achieved the highest initial validation F1-score.

Logistic Regression achieved higher recall.

## 8. Model Optimization

Hyperparameter optimization was performed using temporal expanding-window validation.

Average Precision was used as the tuning metric because the dataset is imbalanced.

The tuned Random Forest achieved the following validation results:

| Metric | Result |
|---|---:|
| Precision | 0.5306 |
| Recall | 0.7647 |
| F1-score | 0.6265 |
| Average Precision | 0.6806 |

### Threshold Selection

- Operational threshold: 0.50
- F1-optimal validation threshold: 0.75

The 0.50 threshold was retained to favor detection of more potentially at-risk students.

The 0.75 threshold achieved a higher validation F1-score, but was not selected as the operational threshold.

## 9. Final Model Evaluation

The final evaluation was performed on prediction period 5, which was kept separate from model tuning.

### Chronological Split

| Dataset | Prediction periods | Records |
|---|---|---:|
| Training | 1–3 | 3,000 |
| Validation | 4 | 1,000 |
| Testing | 5 | 1,000 |

### Final Test Metrics

| Metric | Result |
|---|---:|
| Accuracy | 97.80% |
| Precision | 56.25% |
| Recall | 69.23% |
| F1-score | 62.07% |
| Specificity | 98.56% |
| Average Precision | 72.35% |

### Confusion Matrix

| | Predicted Not At Risk | Predicted At Risk |
|---|---:|---:|
| Actual Not At Risk | 960 | 14 |
| Actual At Risk | 8 | 18 |

The model identified 18 of the 26 actual at-risk test records.

Accuracy should not be interpreted in isolation because the target classes are imbalanced.

## 10. Model Explainability

SHAP and Random Forest feature importance are used to understand model behavior.

### Most Influential Features

| Feature | Importance |
|---|---:|
| Previous score 1 | 0.367682 |
| Average previous score | 0.220592 |
| Previous score 2 | 0.149392 |
| Previous score 3 | 0.123170 |
| Attendance rate | 0.033332 |

The model relies primarily on historical assessment scores.

Individual SHAP explanations show which features push a prediction towards or away from the at-risk class.

SHAP contributions describe model behavior and do not establish causation.

## 11. Responsible AI

The project follows an advisory decision-support approach.

The system provides:

- Predicted risk probability.
- Advisory risk band.
- Influential features.
- Suggested academic support.
- Explicit human-review requirement.

The current advisory categories use provisional probability boundaries:

- Low: below 0.50
- Moderate: 0.50 to below 0.75
- High: 0.75 and above

Support recommendations are based on provisional academic signal thresholds.

These boundaries are not independently validated educational diagnoses.

### Responsible Use Requirements

- Educators must review predictions.
- Predictions must not automatically determine grades.
- Predictions must not determine admissions.
- Predictions must not trigger punishment or disciplinary action.
- Student-level information must be handled privately.
- Model limitations and uncertainty must be considered.

## 12. Streamlit Dashboard

The interactive dashboard provides:

1. Student performance input form.
2. Automatic feature engineering.
3. Saved Random Forest model inference.
4. Risk probability and advisory band.
5. SHAP feature explanations.
6. Feature contribution visualization.
7. Supportive academic recommendations.
8. Human-review disclaimer.

The dashboard loads the trained model through `src/inference/predictor.py`.

### Run the Dashboard

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## 13. Installation and Setup

Clone the repository:

```bash
git clone https://github.com/codewithRahul01/student-performance-risk-prediction.git

cd student-performance-risk-prediction
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 14. Reproduce the Project Pipeline

Run the following commands from the project root, in order:

### Step 1: Generate synthetic data

```bash
python src/data/generate_data.py
```

### Step 2: Preprocess the dataset

```bash
python src/data/preprocess.py
```

### Step 3: Engineer features

```bash
python src/features/build_features.py
```

### Step 4: Create chronological splits

```bash
python src/data/create_splits.py
```

### Step 5: Train baseline models

```bash
python src/models/train_baseline.py
```

### Step 6: Optimize models

```bash
python src/models/optimize_models.py
```

### Step 7: Evaluate the final model

```bash
python src/models/evaluate_final.py
```

### Step 8: Generate explainability reports

```bash
python src/models/explain_model.py
```

### Step 9: Generate responsible AI advisory report

```bash
python src/models/responsible_ai.py
```

The optimization stage generates the saved model artifact required by the dashboard.

**Note:** Generated datasets and model artifacts are excluded from Git by the project's `.gitignore`. Run the pipeline to regenerate them locally before starting the dashboard.

## 15. Automated Testing

The project uses Pytest for automated testing.

Run all tests:

```bash
pytest tests/ -v
```

Generate a test report:

```bash
pytest tests/ -v --junitxml=reports/pytest-results.xml
```

The current test suite contains 18 tests covering:

- Feature engineering calculations.
- Input validation.
- Invalid attendance and assignment values.
- Invalid assessment scores.
- Missing and infinite numerical inputs.
- Model inference.
- Prediction threshold.
- Model artifact availability.
- Prediction reproducibility.
- Leakage prevention.

## 16. Limitations

- The dataset is synthetic and may not represent real student populations.
- The same student cohort appears across multiple prediction periods; the chronological test evaluates future prediction for an existing cohort, not generalization to unseen students.
- Only 26 at-risk observations are present in the final test set, so evaluation metrics have uncertainty.
- The validation period was used for model and threshold selection.
- The model's probability estimates and advisory categories require further calibration and real-world validation.
- SHAP explanations describe model behavior, not causal relationships.
- Support recommendations use provisional rules and require educator judgment.

## 17. Future Improvements

- Evaluate on independently collected, properly licensed educational datasets.
- Assess performance and calibration on new student cohorts.
- Conduct appropriate fairness and robustness evaluations when suitable data and governance are available.
- Improve probability calibration.
- Develop more detailed educator-facing explanations.
- Add secure user authentication and access controls if deployed.
- Validate support recommendations with educational professionals.

---

## 18. Repository

GitHub:

https://github.com/codewithRahul01/student-performance-risk-prediction

**Project:** Student Performance & At-Risk Prediction
**Domain:** School Management
**Category:** Artificial Intelligence and Machine Learning

---

*This project is an educational decision-support prototype. Its predictions are advisory and require human oversight.*
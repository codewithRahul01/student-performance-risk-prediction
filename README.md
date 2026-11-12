
# Student Performance & At-Risk Prediction

## Project Overview

An AI/ML-based academic decision-support system
that identifies students who may be at risk of
scoring below 40 in their next assessment.

The system uses historical academic information
to identify students who may benefit from
additional academic support.

## Dataset

- Total records: 5,000
- Unique students: 1,000
- Prediction periods: 5
- At-risk records: 237
- Not-at-risk records: 4,763

Dataset: Reproducibly generated synthetic data.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Project Phases

1. Project setup and configuration
2. Synthetic dataset generation
3. Exploratory Data Analysis
4. Data Cleaning and Feature Engineering
5. Baseline Model Development
6. Model Optimization
7. Final Model Evaluation
8. Explainability
9. Prediction API
10. Interactive Dashboard
11. Testing and Documentation

## Phase 5 – Model Development

Three classification algorithms were implemented:

1. Dummy Classifier
2. Logistic Regression
3. Random Forest

### Training and Validation Split

- Training: Periods 1–3 (3,000 records)
- Validation: Period 4 (1,000 records)
- Testing: Period 5 (1,000 records)

The test set was not used during Phase 5.

### Initial Validation Results

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Dummy Classifier | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.289 | 0.824 | 0.427 |
| Random Forest | 0.588 | 0.588 | 0.588 |

Random Forest achieved the highest initial
validation F1-score.

Logistic Regression achieved higher recall.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- Average Precision
- Confusion Matrix

## Run the Project

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run data preprocessing:

```bash
python src/data/preprocess.py
```

Run feature engineering:

```bash
python src/features/build_features.py
```

Create chronological splits:

```bash
python src/data/create_splits.py
```

Train baseline models:

```bash
python src/models/train_baseline.py
```

## Responsible AI

This project is designed as an academic-support
decision aid.

Predictions must be reviewed by teachers.

The model must never automatically determine
student grades, admissions or disciplinary actions.

## Limitations

- Dataset contains synthetic academic records.
- The results may not represent real-world
  student populations.
- The validation set contains relatively few
  at-risk examples.
- Model predictions require human review.

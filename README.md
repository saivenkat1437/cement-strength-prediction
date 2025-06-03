# 🧱 Cement Strength Prediction using XGBoost

A complete machine learning pipeline to predict the **strength class of cement-based materials** using features extracted from noisy raw data. The solution includes data cleaning, feature engineering, model training with XGBoost, evaluation, and final prediction generation.

---

## 📌 Problem Statement

In this project, we are given datasets that contain information about cement mix components and need to classify whether the concrete outcome is strong (1) or weak (0). Some of the columns (like `cement_water` and `coarse_fine_aggregate`) contain **two values embedded in a single string**, which must be split and converted into usable features.

---

## 📂 Project Structure

cement-strength-prediction/
│
├── train.csv # Training data/
├── test.csv # Test data for prediction/
├── cement-strength-prediction.py # Main Python script


## 📊 Features Used

After cleaning, the following features are used to train the model:

- `cement`
- `water`
- `slag`
- `fly_ash`
- `plasticizer`
- `coarse_aggregate`
- `fine_aggregate`
- `age`

**Target**: `outcome` (0 = weak, 1 = strong)

---

## 🧹 Data Cleaning Logic

The original columns like `"cement_water"` or `"coarse_fine_aggregate"` are strings with embedded numbers. We use regular expressions to:

1. Extract two float values.
2. Split them into individual columns like `cement`, `water`, `coarse_aggregate`, and `fine_aggregate`.
3. Drop the original mixed columns.

Example:
```python
"400.0 180.0" -> cement: 400.0, water: 180.0

## Model Training
Model: XGBoost Classifier
Preprocessing: StandardScaler for feature normalization
Validation: 80-20 train-validation split
Hyperparameter Tuning: GridSearchCV
Metric: Accuracy & Classification Report

Validation Accuracy: 0.89

Classification Report:
              precision    recall  f1-score   support
           0       0.87      0.90      0.88       140
           1       0.91      0.87      0.89       148
    accuracy                           0.89       288


## Future Enhancements
Replace GridSearchCV with RandomizedSearchCV or Optuna for faster tuning

Add cross-validation metrics

Use SHAP or LIME for interpretability

Build a Streamlit dashboard for live inference


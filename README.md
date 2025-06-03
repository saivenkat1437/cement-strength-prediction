# 🧱 Cement Strength Prediction using XGBoost

This project implements a machine learning pipeline to predict the **strength class of cement mixtures** using chemical and material properties. It demonstrates data cleaning, feature extraction from messy strings, model training with hyperparameter tuning, and generating predictions on unseen test data.

---

## 📋 Problem Statement

Raw data contains mixed string columns such as `'cement_water'` and `'coarse_fine_aggregate'` with two numeric values embedded in a string. The goal is to:

- Extract valid numeric pairs from these columns.
- Use the extracted and other numeric features to predict the `outcome` (strength class) of cement mixtures.
- Train a robust model using XGBoost with hyperparameter tuning.
- Evaluate model performance and generate predictions for test data.

---

## 🔍 Dataset Features

| Feature           | Description                           |
|-------------------|-------------------------------------|
| cement            | Amount of cement                     |
| water             | Amount of water                     |
| slag              | Amount of slag                      |
| fly_ash           | Amount of fly ash                   |
| plasticizer       | Amount of plasticizer               |
| coarse_aggregate   | Amount of coarse aggregate          |
| fine_aggregate     | Amount of fine aggregate            |
| age               | Age of the mixture                  |
| outcome           | Target variable: cement strength class |

---

## 🚀 Project Workflow

### 1. Data Preprocessing

- Load raw CSV files.
- Extract valid float pairs from `'cement_water'` and `'coarse_fine_aggregate'` columns using regex.
- Filter only rows with valid numeric pairs.
- Drop original mixed-string columns after extraction.

### 2. Feature Scaling

- Scale numeric features with `StandardScaler` for normalization.

### 3. Model Training

- Split data into training and validation sets.
- Use `XGBClassifier` with grid search for hyperparameter tuning.
- Train the model on the training data.
- Evaluate performance on validation data.

### 4. Prediction & Submission

- Apply the same cleaning and scaling steps on the test data.
- Generate predictions using the trained model.
- Save predictions in a submission CSV file.

---

## 📊 Performance Metrics (Sample)

## Model Training
Model: XGBoost Classifier
Preprocessing: StandardScaler for feature normalization
Validation: 80-20 train-validation split
Hyperparameter Tuning: GridSearchCV
Metric: Accuracy & Classification Report

Validation Accuracy: 0.89

Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.87      | 0.90   | 0.88     | 140     |
| 1     | 0.91      | 0.87   | 0.89     | 148     |

| Metric  | Value |
|---------|--------|
| Accuracy | 0.89  |
                       


## Future Enhancements
Replace GridSearchCV with RandomizedSearchCV or Optuna for faster tuning

Add cross-validation metrics

Use SHAP or LIME for interpretability

Build a Streamlit dashboard for live inference


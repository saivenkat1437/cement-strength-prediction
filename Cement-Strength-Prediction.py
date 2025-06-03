import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import re

# File paths
train_file = "/Users/saivenkat/Downloads/train.csv"
test_file = "/Users/saivenkat/Downloads/test.csv"
output_file = "/Users/saivenkat/Downloads/submissions.csv"

def extract_valid_rows(df, col, new_cols):
    """
    Extract only rows with two valid float numbers in the string.
    """
    def parse_valid_pair(s):
        if pd.isnull(s):
            return None
        nums = re.findall(r"[\d.]+", s)
        if len(nums) == 2:
            return float(nums[0]), float(nums[1])
        return None

    parsed = df[col].apply(parse_valid_pair)
    mask_valid = parsed.notnull()
    valid_df = df[mask_valid].copy()
    valid_df[new_cols] = pd.DataFrame(parsed[mask_valid].tolist(), index=valid_df.index)
    return valid_df

# Load and clean training data
train_raw = pd.read_csv(train_file)
train_clean = extract_valid_rows(train_raw, 'cement_water', ['cement', 'water'])
train_clean = extract_valid_rows(train_clean, 'coarse_fine_aggregate', ['coarse_aggregate', 'fine_aggregate'])
train = train_clean.drop(columns=['cement_water', 'coarse_fine_aggregate'])

# Features and target
features = ['cement', 'water', 'slag', 'fly_ash', 'plasticizer', 'coarse_aggregate', 'fine_aggregate', 'age']
X = train[features]
y = train['outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
params = {
    'n_estimators': [100],
    'max_depth': [3, 5],
    'learning_rate': [0.1],
    'subsample': [0.8]
}
grid_search = GridSearchCV(xgb, params, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))

# Load and clean test data
test_raw = pd.read_csv(test_file)
test_clean = extract_valid_rows(test_raw, 'cement_water', ['cement', 'water'])
test_clean = extract_valid_rows(test_clean, 'coarse_fine_aggregate', ['coarse_aggregate', 'fine_aggregate'])
test = test_clean.drop(columns=['cement_water', 'coarse_fine_aggregate'])

# Predict
X_test = test[features]
X_test_scaled = scaler.transform(X_test)
test['outcome'] = best_model.predict(X_test_scaled)

# Save results
submission_df = test[['id', 'outcome']]
submission_df.to_csv(output_file, index=False)
print(f"✅ Submission saved as {output_file}")

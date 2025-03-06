import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load the updated dataset
data = pd.read_csv("merged_data_2015_2024.csv")

# Ensure consistent column names (trim spaces if needed)
data.columns = data.columns.str.strip()

# Debug: Print column names to verify
print("Dataset columns:", data.columns)

# Check for required columns
required_columns = ['Player', 'PA', 'Age', 'G', 'AB', 'H', 'HR', 'RBI', 'SB', 'OBP', 'SLG', 'OPS', 'WAR']
missing_cols = [col for col in required_columns if col not in data.columns]
if missing_cols:
    raise KeyError(f"Missing columns in dataset: {missing_cols}")

# Filter the dataset for only hitters (ensure "PA" is valid)
data = data[data['PA'] > 0]

# Select relevant features for prediction
features = ['Age', 'G', 'AB', 'H', 'HR', 'RBI', 'SB', 'OBP', 'SLG', 'OPS']
target = 'WAR'

# Drop rows with missing values
data = data.dropna(subset=features + [target])

# Separate features and target variable
X = data[features] 
y = data[target]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and scaler for later use
import joblib
joblib.dump(model, "HR_predictor_model_V2.pkl")
joblib.dump(scaler, "scaler_HR_V2.pkl")

print("Model training complete. Saved as 'war_predictor_model.pkl'.")


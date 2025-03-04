import pandas as pd
from sklearn.preprocessing import StandardScaler
from pybaseball import batting_stats

# Fetch MLB batting stats for the 2023 season
data = batting_stats(2023)

# Display the first 5 rows to check the data
print("Data before cleaning:")
print(data.head())

# Check for missing values
print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Clean the data: Remove rows with missing values
data_cleaned = data.dropna(subset=['Age', 'G', 'AB', 'H', 'HR', 'WAR'])

# Check for missing values again after cleaning
print("\nMissing values after cleaning:")
print(data_cleaned.isnull().sum())

# Correct data types for specific columns (ensure Age, G, AB, HR, WAR are integers)
data_cleaned['Age'] = data_cleaned['Age'].astype(int)
data_cleaned['G'] = data_cleaned['G'].astype(int)
data_cleaned['AB'] = data_cleaned['AB'].astype(int)
data_cleaned['H'] = data_cleaned['H'].astype(int)
data_cleaned['HR'] = data_cleaned['HR'].astype(int)
data_cleaned['WAR'] = data_cleaned['WAR'].astype(float)  # WAR can have decimals, leave it as float

# Check the first few rows of the cleaned data with correct types
print("\nCleaned data with correct types:")
print(data_cleaned.head())

# Normalize only the columns that should be scaled (WAR is kept for scaling if necessary)
columns_to_scale = ['WAR']
scaler = StandardScaler()

# Ensure the WAR column exists and scale it
if 'WAR' in data_cleaned.columns:
    data_cleaned['WAR'] = scaler.fit_transform(data_cleaned[['WAR']])
else:
    print("Warning: 'WAR' column not found!")

# Check the cleaned data again
print("\nCleaned and scaled WAR data:")
print(data_cleaned.head())

# Save the cleaned data to a CSV file
data_cleaned.to_csv("mlb_player_data_cleaned.csv", index=False)

# Confirm that the file is saved
print("\nData has been saved to 'mlb_player_data_cleaned.csv'.")





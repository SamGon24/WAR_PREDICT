import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("SB_predictor_model.pkl")
scaler = joblib.load("scaler_SB.pkl")

# Load the dataset
data = pd.read_csv("mlb_batting_stats_2015_2024.csv")

# Ensure consistent column names (trim spaces if needed)
data.columns = data.columns.str.strip()

# Check for required columns
required_columns = ['Name', 'Season', 'PA', 'Age', 'G', 'AB', 'H', 'HR', 'RBI', 'SB', 'OBP', 'SLG', 'OPS', 'WAR']
missing_cols = [col for col in required_columns if col not in data.columns]
if missing_cols:
    raise KeyError(f"Missing columns in dataset: {missing_cols}")

# Filter only hitters
data = data[data['PA'] > 0]

# Select relevant features for prediction
features = ['Age', 'G', 'AB', 'H', 'HR', 'RBI', 'SB', 'OBP', 'SLG', 'OPS']

while True:
    # Ask for a player's name
    player_name = input("Enter player's name (or type 'exit' to quit): ").strip()
    
    if player_name.lower() == "exit":
        print("Exiting program.")
        break

    # Find the most recent stats for the player
    player_data = data[data['Name'].str.strip().str.lower() == player_name.lower()]
    
    if player_data.empty:
        print(f"No data found for player: {player_name}. Please try again.")
        continue  # Ask the user again
    else:
        # Use the most recent season
        player_data = player_data.sort_values(by="Season", ascending=False).iloc[0]
        
        # Extract features
        player_features = np.array(player_data[features]).reshape(1, -1)

        # Scale features
        player_scaled = scaler.transform(player_features)

        # Predict SB
        predicted_SB = model.predict(player_scaled)[0]

        print(f"Predicted SBs for {player_name} (based on most recent season): {predicted_SB:.2f}")
    
    # Ask if the user wants to continue
    another = input("Do you want to check another player? (yes/no): ").strip().lower()
    if another != 'yes':
        print("Exiting program.")
        break
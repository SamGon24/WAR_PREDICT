import pandas as pd
import joblib

from features import prepare_prediction_data, FEATURES

# Load model, scaler, and league averages
model = joblib.load("war_predictor_model_INJURY_ADJUSTED.pkl")
scaler = joblib.load("scaler_war_INJURY_ADJUSTED.pkl")
league_avgs = joblib.load("league_avgs_INJURY_ADJUSTED.pkl")

# Load data
data = pd.read_csv("cleaned_dataset_2024_players.csv")
data.columns = data.columns.str.strip()
data = data[data['PA'] > 0]

prediction_data = prepare_prediction_data(data, league_avgs)

# Prediction loop
while True:
    player_name = input("Enter player's name (or 'exit'): ").strip()
    if player_name.lower() == 'exit':
        break
    
    # Exact match for player name
    player = prediction_data[prediction_data['Player'] == player_name]
    
    if player.empty:
        print(f"Player '{player_name}' not found.")
        continue
    
    X = player[FEATURES].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    predicted_war = model.predict(X_scaled)[0]
    print(f"Predicted 2025 WAR: {predicted_war:.2f}\n")
    print(player)


import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load('mlb_war_model.pkl')
scaler = joblib.load('scaler.pkl')

data = pd.read_csv('mlb_hitters_2023.csv')

def predict_war(player_name):
    # Select features
    features = ['Age', 'G', 'AB', 'H', 'HR', 'RBI', 'SB', 'OBP', 'SLG', 'OPS']
    
    # Locate player data
    player_data = data[data['Name'].str.strip().str.lower() == player_name.lower()]
    if player_data.empty:
        print(f"No data found for player: {player_name}")
        return
    
    # Extract player stats and scale them
    player_features = player_data[features]
    player_scaled = scaler.transform(player_features)
    
    # Predict WAR
    predicted_war = model.predict(player_scaled)[0]
    print(f"Predicted WAR for {player_name}: {predicted_war:.2f}")

# Example usage
player_name = "Shohei Ohtani"  # Change this to the player you want to predict
predict_war(player_name)
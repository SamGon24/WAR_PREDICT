import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

# Prediction loop for convenience:
while True:
    player_name = input("Enter player's name (or 'exit'): ").strip()
    if player_name.lower() == 'exit':
        break
    
    player = prediction_data[prediction_data['Player'] == player_name]
    
    if player.empty:
        print(f"Player '{player_name}' not found.")
        continue
    
    X = player[FEATURES].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    predicted_war = model.predict(X_scaled)[0]
    print(f"Predicted 2025 WAR: {predicted_war:.2f}\n")
    
    # Retrieve full WAR history for visualization (EXCLUDING 2020 cause of the PANDEMIC!!!!)
    player_history = data[(data['Player'] == player_name) & (data['Year'] != 2020)][['Year', 'WAR']].sort_values('Year')
    
    # Show hisogram
    plt.figure(figsize=(8, 5))
    plt.plot(player_history['Year'], player_history['WAR'], marker='o', linestyle='-', label='Actual WAR')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    
    injured_seasons = data[(data['Player'] == player_name) & (data['Year'] != 2020) & ((data['G'] < 50) | (data['PA'] < 200))]
    plt.scatter(injured_seasons['Year'], injured_seasons['WAR'], color='red', label='Injury Season', zorder=3)
    
    plt.scatter([2025], [predicted_war], color='green', marker='D', s=100, label='Predicted 2025 WAR')
    
    plt.xlabel("Year")
    plt.ylabel("WAR")
    plt.title(f"WAR Evolution for {player_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

import pandas as pd
import joblib

# Load NEW model and scaler
model = joblib.load("war_predictor_model_ORIGINAL.pkl")
scaler = joblib.load("scaler_war_ORIGINAL.pkl")

# Load data
data = pd.read_csv("sorted_dataset.csv")
data.columns = data.columns.str.strip()
data = data[data['PA'] > 0]

def prepare_prediction_data(df):
    # Career totals
    career_stats = df.groupby('Player').agg(
        Career_G=('G', 'sum'),
        Career_HR=('HR', 'sum'),
        Career_SB=('SB', 'sum')
    ).reset_index()
    
    # 3-year averages
    three_year_avg = df.groupby('Player').apply(
        lambda x: x.tail(3)[['OBP', 'SLG']].mean()
    ).reset_index()
    three_year_avg.columns = ['Player', 'OBP_3yr', 'SLG_3yr']
    
    # Latest season data (2024)
    latest_data = df.sort_values(['Player', 'Year']).groupby('Player').last().reset_index()
    latest_data = latest_data.merge(career_stats, on='Player')
    latest_data = latest_data.merge(three_year_avg, on='Player')
    
    # Age for next season (2025)
    latest_data['Age'] = latest_data['Age'] + 1
    
    return latest_data[['Player', 'Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR']]

prediction_data = prepare_prediction_data(data)

# Prediction loop
while True:
    player_name = input("Enter player's name (or 'exit'): ").strip().lower()
    if player_name == 'exit':
        break
    
    player = prediction_data[prediction_data['Player'].str.lower() == player_name]
    
    if player.empty:
        print(f"Player {player_name} not found.")
        continue
    
    # Ensure feature order matches training
    features = ['Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR']
    X = player[features].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    predicted_war = model.predict(X_scaled)[0]
    print(f"Predicted 2025 WAR: {predicted_war:.2f}\n")
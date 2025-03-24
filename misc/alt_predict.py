import pandas as pd
import joblib

# Load model, scaler, and league averages
model = joblib.load("war_predictor_model_CORRECTED.pkl")
scaler = joblib.load("scaler_war_CORRECTED.pkl")
league_avgs = joblib.load("league_avgs_CORRECTED.pkl")

# Load data
data = pd.read_csv("sorted_dataset.csv")
data.columns = data.columns.str.strip()
data = data[data['PA'] > 0]

def prepare_prediction_data(df):
    # Calculate cumulative stats
    career_stats = df.groupby('Player').agg(
        Career_G=('G', 'sum'),
        Career_HR=('HR', 'sum'),
        Career_SB=('SB', 'sum')
    ).reset_index()
    
    # 3-year averages (use last 3 seasons)
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
    
    # Mark rookies (Career_G = current season's G)
    latest_data['Is_Rookie'] = (latest_data['Career_G'] == latest_data['G']).astype(int)
    
    # Impute missing data for rookies
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'OBP_3yr'] = league_avgs['league_avg_obp']
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'SLG_3yr'] = league_avgs['league_avg_slg']
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'Career_G'] = 0
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'Career_HR'] = 0
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'Career_SB'] = 0
    
    return latest_data[['Player', 'Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR', 'Is_Rookie']]

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
    features = ['Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR', 'Is_Rookie']
    X = player[features].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    predicted_war = model.predict(X_scaled)[0]
    print(f"Predicted 2025 WAR: {predicted_war:.2f}\n")
import pandas as pd
import joblib

# Load model, scaler, and league averages
model = joblib.load("war_predictor_model_INJURY_ADJUSTED.pkl")
scaler = joblib.load("scaler_war_INJURY_ADJUSTED.pkl")
league_avgs = joblib.load("league_avgs_INJURY_ADJUSTED.pkl")

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
    
    # Detect injured seasons (G < 50 or PA < 200)
    df['Injured_Season'] = ((df['G'] < 50) | (df['PA'] < 200)).astype(int)
    
    # Calculate 3-year rolling averages, excluding injured seasons
    df['OBP_healthy'] = df['OBP'].where(df['Injured_Season'] == 0)
    df['SLG_healthy'] = df['SLG'].where(df['Injured_Season'] == 0)
    
    df['OBP_3yr'] = df.groupby('Player')['OBP_healthy'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['SLG_3yr'] = df.groupby('Player')['SLG_healthy'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    # If no valid data, use career average
    df['OBP_3yr'] = df.groupby('Player')['OBP_3yr'].transform(
        lambda x: x.fillna(x.mean())
    )
    df['SLG_3yr'] = df.groupby('Player')['SLG_3yr'].transform(
        lambda x: x.fillna(x.mean())
    )
    
    # Latest season data (2024)
    latest_data = df.sort_values(['Player', 'Year']).groupby('Player').last().reset_index()
    latest_data = latest_data.merge(career_stats, on='Player')
    
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
    
    return latest_data[['Player', 'Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR', 'Is_Rookie', 'Injured_Season']]

prediction_data = prepare_prediction_data(data)

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
    
    # Ensure feature order matches training
    features = ['Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR', 'Is_Rookie', 'Injured_Season']
    X = player[features].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    predicted_war = model.predict(X_scaled)[0]
    print(f"Predicted 2025 WAR: {predicted_war:.2f}\n")
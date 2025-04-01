import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model, scaler, and league averages
model = joblib.load("war_predictor_model_INJURY_ADJUSTED.pkl")
scaler = joblib.load("scaler_war_INJURY_ADJUSTED.pkl")
league_avgs = joblib.load("league_avgs_INJURY_ADJUSTED.pkl")

# Load data
data = pd.read_csv("cleaned_dataset_2024_players.csv")
data.columns = data.columns.str.strip()
data = data[data['PA'] > 0]

def prepare_prediction_data(df):
    career_stats = df.groupby('Player').agg(
        Career_G=('G', 'sum'),
        Career_HR=('HR', 'sum'),
        Career_SB=('SB', 'sum')
    ).reset_index()
    
    df['Injured_Season'] = ((df['G'] < 50) | (df['PA'] < 200)).astype(int)
    
    three_year_avg = df.groupby('Player').apply(
        lambda x: x[~x['Injured_Season'].astype(bool)].tail(3)[['OBP', 'SLG']].mean()
    ).reset_index()
    three_year_avg.columns = ['Player', 'OBP_3yr', 'SLG_3yr']

    latest_data = df.sort_values(['Player', 'Year']).groupby('Player').last().reset_index()
    
    injured_players_2024 = latest_data[(latest_data['Year'] == 2024) & ((latest_data['G'] < 50) | (latest_data['PA'] < 200))]['Player']
    last_valid_season = df[df['Player'].isin(injured_players_2024) & (df['Year'] == 2023)]
    latest_data.loc[latest_data['Player'].isin(injured_players_2024)] = last_valid_season.set_index('Player').reindex(latest_data['Player'].values).reset_index()
    
    latest_data = latest_data.merge(career_stats, on='Player')
    latest_data = latest_data.merge(three_year_avg, on='Player')
    
    latest_data['Age'] = latest_data['Age'] + 1
    latest_data['Is_Rookie'] = (latest_data['Career_G'] == latest_data['G']).astype(int)
    
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'OBP_3yr'] = league_avgs['league_avg_obp']
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'SLG_3yr'] = league_avgs['league_avg_slg']
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'Career_G'] = 0
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'Career_HR'] = 0
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'Career_SB'] = 0
    
    latest_data['Injured_Season'] = ((latest_data['G'] < 50) | (latest_data['PA'] < 200)).astype(int)
    
    return latest_data[['Player', 'Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR', 'Is_Rookie', 'Injured_Season']]

prediction_data = prepare_prediction_data(data)

# Prediction loop
while True:
    player_name = input("Enter player's name (or 'exit'): ").strip()
    if player_name.lower() == 'exit':
        break
    
    player = prediction_data[prediction_data['Player'] == player_name]
    
    if player.empty:
        print(f"Player '{player_name}' not found.")
        continue
    
    features = ['Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR', 'Is_Rookie', 'Injured_Season']
    X = player[features].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    predicted_war = model.predict(X_scaled)[0]
    print(f"Predicted 2025 WAR: {predicted_war:.2f}\n")
    
    # Retrieve full WAR history for visualization
    player_history = data[data['Player'] == player_name][['Year', 'WAR']].sort_values('Year')
    
    plt.figure(figsize=(8, 5))
    plt.plot(player_history['Year'], player_history['WAR'], marker='o', linestyle='-', label='Actual WAR')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    
    injured_seasons = data[(data['Player'] == player_name) & ((data['G'] < 50) | (data['PA'] < 200))]
    plt.scatter(injured_seasons['Year'], injured_seasons['WAR'], color='red', label='Injury Season', zorder=3)
    
    plt.scatter([2025], [predicted_war], color='green', marker='D', s=100, label='Predicted 2025 WAR')
    
    plt.xlabel("Year")
    plt.ylabel("WAR")
    plt.title(f"WAR Evolution for {player_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
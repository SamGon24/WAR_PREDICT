import pandas as pd
import joblib
from fuzzywuzzy import process

# Load model and scaler
model = joblib.load("war_predictor_final.pkl")
scaler = joblib.load("scaler_war_final.pkl")

# Load data
data = pd.read_csv("cleaned_dataset.csv")
data.columns = data.columns.str.strip()

# Convert all numeric columns to float and handle NaNs
numeric_cols = ['AB', 'H', 'HR', 'SB', 'G', 'OBP', 'SLG', 'WAR', 'Age']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# Filter hitters
data = data[data['PA'] > 0]

# Feature engineering (same as in training)
def create_features(df):
    # Exclude injury seasons (AB < 200) from calculations
    valid_seasons = df[df['AB'] >= 200].copy()
    
    # Cumulative stats (only valid seasons)
    valid_seasons['Career_G'] = valid_seasons.groupby('Player')['G'].cumsum()
    valid_seasons['Career_HR'] = valid_seasons.groupby('Player')['HR'].cumsum()
    valid_seasons['Career_SB'] = valid_seasons.groupby('Player')['SB'].cumsum()
    
    # Merge back cumulative stats to the original dataframe
    df = df.merge(
        valid_seasons[['Player', 'Year', 'Career_G', 'Career_HR', 'Career_SB']],
        on=['Player', 'Year'],
        how='left'
    )
    
    # Forward-fill cumulative stats to handle gaps
    df['Career_G'] = df.groupby('Player')['Career_G'].ffill().fillna(0)
    df['Career_HR'] = df.groupby('Player')['Career_HR'].ffill().fillna(0)
    df['Career_SB'] = df.groupby('Player')['Career_SB'].ffill().fillna(0)
    
    # Rolling averages (only valid seasons)
    df['OBP_3yr'] = df[df['AB'] >= 200].groupby('Player')['OBP'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['SLG_3yr'] = df[df['AB'] >= 200].groupby('Player')['SLG'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    # Injury-related features
    df['Injured'] = (df['AB'] < 200).astype(int)
    df['Post_Injury'] = df.groupby('Player')['Injured'].shift(1).fillna(0)  # 1 if previous season was injured
    
    # Age adjustment for next season
    df['Age'] = df['Age'] + 1
    
    # Peak performance features
    df['Peak_WAR'] = df.groupby('Player')['WAR'].transform('max')
    df['WAR_percentile'] = df.groupby('Year')['WAR'].transform(
        lambda x: x.rank(pct=True)
    )
    
    return df

def prepare_player_data(df, player_name):
    # Filter data for the specified player
    player_data = df[df['Player'] == player_name].sort_values('Year')
    
    # Debug: Print raw data for the player
    print(f"\nRaw data for {player_name}:")
    print(player_data)
    
    # If the player missed the latest season (e.g., 2024), impute stats based on their last full season
    if player_data.iloc[-1]['AB'] < 200:  # Injury season
        latest_full_season = player_data[player_data['AB'] >= 200].iloc[-1].to_dict()
        imputed_season = {
            'Year': player_data.iloc[-1]['Year'] + 1,
            'Age': latest_full_season['Age'] + 1,
            'AB': latest_full_season['AB'] * 0.95,  # 5% decline (league average)
            'H': latest_full_season['H'] * 0.95,
            'HR': latest_full_season['HR'] * 0.95,
            'SB': latest_full_season['SB'] * 0.95,
            'OBP': latest_full_season['OBP'] * 0.97,  # Smaller decline for OBP/SLG
            'SLG': latest_full_season['SLG'] * 0.97,
            'WAR': latest_full_season['WAR'] * 0.95
        }
        player_data = pd.concat([player_data, pd.DataFrame([imputed_season])], ignore_index=True)
    
    # Apply feature engineering
    player_data = create_features(player_data)
    
    # Debug: Print the player's features
    print(f"\nFeatures for {player_name}:")
    print(player_data.tail(1)[features])
    
    # Return the latest row for prediction
    return player_data.tail(1)[features]

# Define features (must match the training script)
features = [
    'Age', 'Career_G', 'Career_HR', 'Career_SB',
    'OBP_3yr', 'SLG_3yr', 'WAR', 'Post_Injury',
    'Peak_WAR', 'WAR_percentile'
]

# Fuzzy matching function
def find_closest_player(player_name, player_list):
    # Use fuzzywuzzy to find the closest match
    closest_match, score = process.extractOne(player_name, player_list)
    return closest_match, score

# Get list of all players in the dataset
all_players = data['Player'].unique()

# Prediction loop
while True:
    player_name = input("Enter player's name (or 'exit'): ").strip()
    if player_name.lower() == 'exit':
        break
    
    # Find the closest match using fuzzy matching
    closest_match, score = find_closest_player(player_name, all_players)
    
    # If the match score is too low, assume the player doesn't exist
    if score < 70:  # Adjust threshold as needed
        print(f"Player '{player_name}' not found. Did you mean '{closest_match}'?")
        continue
    
    # Confirm the match with the user
    confirm = input(f"Did you mean '{closest_match}'? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Please try again.")
        continue
    
    # Prepare player data
    player_features = prepare_player_data(data, closest_match)
    if player_features.empty:
        print(f"Player '{closest_match}' not found in the dataset.")
        continue
    
    # Scale features and predict
    player_features_scaled = scaler.transform(player_features)
    predicted_war = model.predict(player_features_scaled)[0]
    print(f"Predicted 2025 WAR for {closest_match}: {predicted_war:.2f}\n")
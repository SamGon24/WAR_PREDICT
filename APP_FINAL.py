from flask import Flask, render_template, request, send_from_directory
from markupsafe import Markup
import pandas as pd
import joblib
import difflib
import os
import random

app = Flask(__name__)

# Configuration
app.config['PLAYER_IMAGES'] = 'static/player_images'

# Load model, scaler, and league averages
model = joblib.load("war_predictor_model_INJURY_ADJUSTED.pkl")
scaler = joblib.load("scaler_war_INJURY_ADJUSTED.pkl")
league_avgs = joblib.load("league_avgs_INJURY_ADJUSTED.pkl")

# Load data
data = pd.read_csv("cleaned_dataset_2024_players.csv")
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
    
    # 3-year averages (exclude injured seasons)
    three_year_avg = df.groupby('Player').apply(
        lambda x: x[~x['Injured_Season'].astype(bool)].tail(3)[['OBP', 'SLG']].mean()
    ).reset_index()
    three_year_avg.columns = ['Player', 'OBP_3yr', 'SLG_3yr']

    # Latest season data (2024)
    latest_data = df.sort_values(['Player', 'Year']).groupby('Player').last().reset_index()
    
    # Identify injured players in 2024
    injured_players_2024 = latest_data[(latest_data['Year'] == 2024) & ((latest_data['G'] < 50) | (latest_data['PA'] < 200))]['Player']
    
    # Get last valid season (2023) for injured players
    last_valid_season = df[df['Player'].isin(injured_players_2024) & (df['Year'] == 2023)]
    
    # Replace injured 2024 season with 2023 stats
    latest_data.loc[latest_data['Player'].isin(injured_players_2024)] = last_valid_season.set_index('Player').reindex(latest_data['Player'].values).reset_index()
    
    # Merge career stats and 3-year averages
    latest_data = latest_data.merge(career_stats, on='Player')
    latest_data = latest_data.merge(three_year_avg, on='Player')
    
    # Update age for 2025
    latest_data['Age'] = latest_data['Age'] + 1
    
    # Mark rookies (Career_G = current season's G)
    latest_data['Is_Rookie'] = (latest_data['Career_G'] == latest_data['G']).astype(int)
    
    # Impute missing data for rookies
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'OBP_3yr'] = league_avgs['league_avg_obp']
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'SLG_3yr'] = league_avgs['league_avg_slg']
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'Career_G'] = 0
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'Career_HR'] = 0
    latest_data.loc[latest_data['Is_Rookie'] == 1, 'Career_SB'] = 0
    
    # Detect injured season (G < 50 or PA < 200)
    latest_data['Injured_Season'] = ((latest_data['G'] < 50) | (latest_data['PA'] < 200)).astype(int)
    
    return latest_data[['Player', 'Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR', 'Is_Rookie', 'Injured_Season']]

def get_player_image_path(player_name):
    """Check if player image exists and return the path if found, None otherwise"""
    # Standardize filename
    base_name = player_name.lower().replace(' ', '_').replace('.', '').replace("'", '')
    variations = [
        f"{base_name}.jpg",  # First try full name (mike_trout.jpg)
        f"{base_name[0]}_{base_name.split('_')[-1]}.jpg"  # Then try first initial (m_trout.jpg)
    ]
    
    for filename in variations:
        image_path = os.path.join(app.config['PLAYER_IMAGES'], filename)
        if os.path.exists(image_path):
            return f"/static/player_images/{filename}"
    return None

def get_players_with_images():
    """Return list of players who have images available"""
    players_with_images = []
    for player in prediction_data['Player'].unique():
        if get_player_image_path(player):
            players_with_images.append(player)
    return players_with_images

# Prepare player data
prediction_data = prepare_prediction_data(data)

@app.context_processor
def utility_processor():
    return dict(get_player_image=get_player_image_path)

@app.route("/", methods=["GET", "POST"])
def index():
    players_with_images = get_players_with_images()
    
    if request.method == "POST":
        player_name = request.form["player_name"].strip().lower()
        player = prediction_data[prediction_data['Player'].str.lower() == player_name]

        if player.empty:
            close_matches = difflib.get_close_matches(
                player_name, 
                [p.lower() for p in players_with_images], 
                n=1, 
                cutoff=0.6
            )
            
            if close_matches:
                suggested_name = close_matches[0].title()
                suggestion_html = f"""
                <form method='post'>
                    <input type='hidden' name='player_name' value='{suggested_name}'>
                    <button type='submit' class='btn btn-link p-0'>{suggested_name}</button>
                </form>
                """
                return render_template(
                    "index.html", 
                    error=Markup(f"Player '{player_name.title()}' not found. Did you mean {suggestion_html}?"),
                    suggested_players=random.sample(players_with_images, min(5, len(players_with_images)))
                )
            
            return render_template(
                "index.html", 
                error=f"Player '{player_name.title()}' not found.",
                suggested_players=random.sample(players_with_images, min(5, len(players_with_images)))
            )

        # Prepare input data for prediction
        features = ['Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR', 'Is_Rookie', 'Injured_Season']
        X = player[features].values.reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Predict WAR
        predicted_war = model.predict(X_scaled)[0]
        
        # Get player image
        player_display_name = player.iloc[0]['Player']
        player_image = get_player_image_path(player_display_name)

        return render_template(
            "result.html", 
            player=player_display_name,
            war=round(predicted_war, 2),
            player_image=player_image
        )

    # GET request - show form with 5 random suggestions (only players with images)
    return render_template(
        "index.html",
        suggested_players=random.sample(players_with_images, min(5, len(players_with_images)))
    )

@app.route('/player_images/<filename>')
def serve_player_image(filename):
    return send_from_directory(app.config['PLAYER_IMAGES'], filename)

if __name__ == "__main__":
    os.makedirs(app.config['PLAYER_IMAGES'], exist_ok=True)
    app.run(debug=True)



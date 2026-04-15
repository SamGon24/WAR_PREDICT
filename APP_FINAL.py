from flask import Flask, render_template, request, send_from_directory
from markupsafe import Markup
import pandas as pd
import joblib
import difflib
import os
import random

from features import prepare_prediction_data, FEATURES

app = Flask(__name__)

# Configuration for the player images
app.config['PLAYER_IMAGES'] = 'static/player_images'

# Load model, scaler, and league averages
model = joblib.load("war_predictor_model_INJURY_ADJUSTED.pkl")
scaler = joblib.load("scaler_war_INJURY_ADJUSTED.pkl")
league_avgs = joblib.load("league_avgs_INJURY_ADJUSTED.pkl")

# Load data
data = pd.read_csv("cleaned_dataset_2024_players.csv")
data.columns = data.columns.str.strip()
data = data[data['PA'] > 0]


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
prediction_data = prepare_prediction_data(data, league_avgs)

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
        X = player[FEATURES].values.reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Predict WAR
        predicted_war = model.predict(X_scaled)[0]
        
        # Get x player image
        player_display_name = player.iloc[0]['Player']
        player_image = get_player_image_path(player_display_name)

        return render_template(
            "result.html", 
            player=player_display_name,
            war=round(predicted_war, 2),
            player_image=player_image
        )

    # GET request show form with 5 random suggestions only players with images. I dont have all of them
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


from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("war_predictor_model_ORIGINAL.pkl")
scaler = joblib.load("scaler_war_ORIGINAL.pkl")

# Load data
data = pd.read_csv("sorted_dataset.csv")
data.columns = data.columns.str.strip()
data = data[data['PA'] > 0]

def prepare_prediction_data(df):
    career_stats = df.groupby('Player').agg(
        Career_G=('G', 'sum'),
        Career_HR=('HR', 'sum'),
        Career_SB=('SB', 'sum')
    ).reset_index()

    three_year_avg = df.groupby('Player').apply(
        lambda x: x.tail(3)[['OBP', 'SLG']].mean()
    ).reset_index()
    three_year_avg.columns = ['Player', 'OBP_3yr', 'SLG_3yr']

    latest_data = df.sort_values(['Player', 'Year']).groupby('Player').last().reset_index()
    latest_data = latest_data.merge(career_stats, on='Player')
    latest_data = latest_data.merge(three_year_avg, on='Player')

    latest_data['Age'] = latest_data['Age'] + 1

    return latest_data[['Player', 'Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR']]

prediction_data = prepare_prediction_data(data)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        player_name = request.form["player_name"].strip().lower()
        player = prediction_data[prediction_data['Player'].str.lower() == player_name]

        if player.empty:
            return render_template("index.html", error=f"Player '{player_name}' not found.")

        features = ['Age', 'Career_G', 'Career_HR', 'Career_SB', 'OBP_3yr', 'SLG_3yr', 'WAR']
        X = player[features].values.reshape(1, -1)
        X_scaled = scaler.transform(X)

        predicted_war = model.predict(X_scaled)[0]

        return render_template("result.html", player=player_name.title(), war=round(predicted_war, 2))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
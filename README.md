# WAR Predict

A machine learning web app that predicts a MLB hitter's **Wins Above Replacement (WAR)** for the 2025 season.

Live at: **https://web-production-1a459.up.railway.app/**

---

## How it works

A **Random Forest Regressor** is trained on multi-year batting data (2015–2024) sourced from [Baseball Reference](https://www.baseball-reference.com/). The model uses injury-adjusted features to account for shortened seasons:

| Feature | Description |
|---|---|
| `Age` | Projected age for the upcoming season |
| `Career_G / HR / SB` | Cumulative career games, home runs, stolen bases |
| `OBP_3yr / SLG_3yr` | 3-year rolling OBP and SLG, excluding injured seasons |
| `WAR` | Most recent healthy season WAR |
| `Is_Rookie` | Flag for first-year players (imputed with league averages) |
| `Injured_Season` | Flag if the player's last season was injury-shortened (G < 50 or PA < 200) |

Only hitters are supported. Pitchers require a separate model.

---

## Project structure

```
app.py              # Flask web application (entry point)
train.py            # Model training script
predict.py          # CLI prediction loop
predict_chart.py    # CLI prediction loop with WAR history chart
features.py         # Shared feature engineering (imported by all scripts)
Procfile            # Railway/Heroku deployment config
requirements.txt    # Production dependencies
templates/          # HTML templates (index, result)
static/             # Logo and player headshot images
```

---

## Running locally

```bash
# Activate your virtual environment
source venv/bin/activate

# Start the web app
python app.py
```

## Retraining the model

```bash
python train.py
```

This reads `sorted_dataset.csv`, retrains the Random Forest, and overwrites the model artifacts (`*.pkl`).

## CLI prediction (no browser)

```bash
python predict.py            # text output only
python predict_chart.py      # includes a WAR history line chart
```

---

## Data source

[Baseball Reference](https://www.baseball-reference.com/) via [pybaseball](https://github.com/jldbc/pybaseball).

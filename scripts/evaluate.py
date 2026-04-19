"""
Model evaluation script.

Trains both a baseline model (original 9 features) and the expanded
model (current feature set) on the same data, evaluates both on the
2023 held-out test set, and produces two plots:

  1. RMSE / MAE / R² comparison table printed to stdout
  2. Feature importance bar chart for the expanded model
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from features import FEATURES

# ---------------------------------------------------------------------------
# Data preparation (mirrors train.py exactly)
# ---------------------------------------------------------------------------

data = pd.read_csv("sorted_dataset.csv")
data.columns = data.columns.str.strip()
data = data[data['PA'] > 0]
data = data.sort_values(['Player', 'Year'])
data['Next_WAR'] = data.groupby('Player')['WAR'].shift(-1)
data = data.dropna(subset=['Next_WAR'])

train_raw = data[data['Year'] < 2023]
test_raw  = data[data['Year'] == 2023]

league_avg_obp     = train_raw['OBP'].mean()
league_avg_slg     = train_raw['SLG'].mean()
league_avg_bb_rate = (train_raw['BB'] / train_raw['PA']).mean()
league_avg_k_rate  = (train_raw['SO'] / train_raw['PA']).mean()


def create_features(df):
    df['Career_G']  = df.groupby('Player')['G'].cumsum()
    df['Career_HR'] = df.groupby('Player')['HR'].cumsum()
    df['Career_SB'] = df.groupby('Player')['SB'].cumsum()

    df['Injured_Season'] = ((df['G'] < 50) | (df['PA'] < 200)).astype(int)

    df['OBP_healthy'] = df['OBP'].where(df['Injured_Season'] == 0)
    df['SLG_healthy'] = df['SLG'].where(df['Injured_Season'] == 0)
    df['OBP_3yr'] = df.groupby('Player')['OBP_healthy'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['SLG_3yr'] = df.groupby('Player')['SLG_healthy'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['OBP_3yr'] = df.groupby('Player')['OBP_3yr'].transform(lambda x: x.fillna(x.mean()))
    df['SLG_3yr'] = df.groupby('Player')['SLG_3yr'].transform(lambda x: x.fillna(x.mean()))

    df['BB_rate'] = df['BB'] / df['PA']
    df['K_rate']  = df['SO'] / df['PA']

    df['OPS_plus_healthy'] = df['OPS+'].where(df['Injured_Season'] == 0)
    df['OPS_plus_3yr'] = df.groupby('Player')['OPS_plus_healthy'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['OPS_plus_3yr'] = df.groupby('Player')['OPS_plus_3yr'].transform(lambda x: x.fillna(x.mean()))

    df['WAR_trend'] = (df['WAR'] - df.groupby('Player')['WAR'].shift(2)).fillna(0)
    df['Peak_WAR']  = df.groupby('Player')['WAR'].transform(lambda x: x.expanding().max())

    df['Age'] = df['Age'] + 1
    df['Is_Rookie'] = df.groupby('Player')['Year'].transform(lambda x: (x == x.min()).astype(int))

    df.loc[df['Is_Rookie'] == 1, 'OBP_3yr']  = league_avg_obp
    df.loc[df['Is_Rookie'] == 1, 'SLG_3yr']  = league_avg_slg
    df.loc[df['Is_Rookie'] == 1, 'BB_rate']  = league_avg_bb_rate
    df.loc[df['Is_Rookie'] == 1, 'K_rate']   = league_avg_k_rate
    df.loc[df['Is_Rookie'] == 1, 'WAR_trend'] = 0
    df.loc[df['Is_Rookie'] == 1, 'Career_G']  = 0
    df.loc[df['Is_Rookie'] == 1, 'Career_HR'] = 0
    df.loc[df['Is_Rookie'] == 1, 'Career_SB'] = 0

    return df


train = train_raw.groupby('Player', group_keys=False).apply(create_features)
test  = test_raw.groupby('Player',  group_keys=False).apply(create_features)

BASELINE_FEATURES = [
    'Age', 'Career_G', 'Career_HR', 'Career_SB',
    'OBP_3yr', 'SLG_3yr', 'WAR', 'Is_Rookie', 'Injured_Season',
]
target = 'Next_WAR'

train = train.dropna(subset=FEATURES + [target])
test  = test.dropna(subset=FEATURES + [target])


# ---------------------------------------------------------------------------
# Train and evaluate both models
# ---------------------------------------------------------------------------

def train_and_eval(feature_list, label):
    X_tr = train[feature_list].values
    y_tr = train[target].values
    X_te = test[feature_list].values
    y_te = test[target].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_tr_s, y_tr)
    preds = model.predict(X_te_s)

    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mae  = mean_absolute_error(y_te, preds)
    r2   = r2_score(y_te, preds)

    print(f"\n{label}")
    print(f"  Features : {len(feature_list)}")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  R²       : {r2:.4f}")

    return model, scaler, preds, y_te


print("=" * 45)
print("  WAR Model Evaluation — 2023 Test Set")
print("=" * 45)

baseline_model, _, baseline_preds, y_te = train_and_eval(BASELINE_FEATURES, "Baseline (9 features)")
expanded_model, _, expanded_preds, _    = train_and_eval(FEATURES,           "Expanded (14 features)")

rmse_delta = (
    np.sqrt(mean_squared_error(y_te, baseline_preds)) -
    np.sqrt(mean_squared_error(y_te, expanded_preds))
)
print(f"\n  RMSE improvement: {rmse_delta:+.4f} WAR")
print("=" * 45)


# ---------------------------------------------------------------------------
# Feature importance chart for the expanded model
# ---------------------------------------------------------------------------

importances = expanded_model.feature_importances_
indices = np.argsort(importances)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(
    [FEATURES[i] for i in indices],
    importances[indices],
    color='steelblue',
)
ax.set_xlabel("Importance")
ax.set_title("Feature Importances — Expanded Model")
plt.tight_layout()
plt.show()

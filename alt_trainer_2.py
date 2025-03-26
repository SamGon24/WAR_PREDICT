import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = pd.read_csv("sorted_dataset.csv")
data.columns = data.columns.str.strip()

# Filter hitters (dataset is already clean, but just in case...)
data = data[data['PA'] > 0]

# Sort chronologically
data = data.sort_values(['Player', 'Year'])

# Create next season's WAR target
data['Next_WAR'] = data.groupby('Player')['WAR'].shift(-1)
data = data.dropna(subset=['Next_WAR'])

# Split data into train/test FIRST to avoid leakage
train = data[data['Year'] < 2023]
test = data[data['Year'] == 2023]

# Compute league averages from TRAINING DATA ONLY
league_avg_obp = train['OBP'].mean()
league_avg_slg = train['SLG'].mean()

def create_features(df):
    # Calculate cumulative stats
    df['Career_G'] = df.groupby('Player')['G'].cumsum()
    df['Career_HR'] = df.groupby('Player')['HR'].cumsum()
    df['Career_SB'] = df.groupby('Player')['SB'].cumsum()
    
    # Detect injured seasons (G < 50 or PA < 200)
    df['Injured_Season'] = ((df['G'] < 50) | (df['PA'] < 200)).astype(int)
    
    # Calculate 3-year rolling averages, excluding injured seasons
    # Use .where() to mask injured seasons as NaN
    df['OBP_healthy'] = df['OBP'].where(df['Injured_Season'] == 0)
    df['SLG_healthy'] = df['SLG'].where(df['Injured_Season'] == 0)
    
    # Rolling average with min_periods=1 (allow partial windows)
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
    
    # Age for next season
    df['Age'] = df['Age'] + 1
    
    # Mark rookies (first MLB season)
    df['Is_Rookie'] = df.groupby('Player')['Year'].transform(
        lambda x: (x == x.min()).astype(int)
    )
    
    # Impute missing data for rookies
    df.loc[df['Is_Rookie'] == 1, 'OBP_3yr'] = league_avg_obp
    df.loc[df['Is_Rookie'] == 1, 'SLG_3yr'] = league_avg_slg
    df.loc[df['Is_Rookie'] == 1, 'Career_G'] = 0
    df.loc[df['Is_Rookie'] == 1, 'Career_HR'] = 0
    df.loc[df['Is_Rookie'] == 1, 'Career_SB'] = 0
    
    return df

# Apply feature engineering to train/test
train = train.groupby('Player', group_keys=False).apply(create_features)
test = test.groupby('Player', group_keys=False).apply(create_features)

# Final features (add Is_Rookie and Injured_Season)
features = [
    'Age', 'Career_G', 'Career_HR', 'Career_SB',
    'OBP_3yr', 'SLG_3yr', 'WAR', 'Is_Rookie', 'Injured_Season'
]
target = 'Next_WAR'

# Filter valid data
train = train.dropna(subset=features + [target])
test = test.dropna(subset=features + [target])

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Save artifacts
joblib.dump(model, "war_predictor_model_INJURY_ADJUSTED.pkl")
joblib.dump(scaler, "scaler_war_INJURY_ADJUSTED.pkl")
joblib.dump(
    {'league_avg_obp': league_avg_obp, 'league_avg_slg': league_avg_slg},
    "league_avgs_INJURY_ADJUSTED.pkl"
)

print("Model trained successfully with injury adjustment!")
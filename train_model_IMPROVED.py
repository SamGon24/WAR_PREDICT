import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = pd.read_csv("merged_data_2015_2024.csv")
data.columns = data.columns.str.strip()

# Filter hitters
data = data[data['PA'] > 0]

# Sort chronologically
data = data.sort_values(['Player', 'Year'])

# Create next season's WAR target
data['Next_WAR'] = data.groupby('Player')['WAR'].shift(-1)
data = data.dropna(subset=['Next_WAR'])

# Feature engineering
def create_features(df):
    df['Career_G'] = df.groupby('Player')['G'].cumsum()
    df['Career_HR'] = df.groupby('Player')['HR'].cumsum()
    df['Career_SB'] = df.groupby('Player')['SB'].cumsum()
    
    # Calculate 3-year rolling averages
    df['OBP_3yr'] = df.groupby('Player')['OBP'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['SLG_3yr'] = df.groupby('Player')['SLG'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    # Age for next season
    df['Age'] = df['Age'] + 1
    
    return df

data = data.groupby('Player', group_keys=False).apply(create_features)

# Final features
features = [
    'Age', 'Career_G', 'Career_HR', 'Career_SB',
    'OBP_3yr', 'SLG_3yr', 'WAR'
]
target = 'Next_WAR'

# Filter valid data
data = data.dropna(subset=features + [target])

# Split data
train = data[data['Year'] < 2023]
test = data[data['Year'] == 2023]

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
joblib.dump(model, "war_predictor_model_IMPROVED.pkl")
joblib.dump(scaler, "scaler_war_IMPROVED.pkl")

print("Model trained successfully!")






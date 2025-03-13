import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = pd.read_csv("sorted_dataset.csv")
data.columns = data.columns.str.strip()

# Convert all numeric columns to float and handle NaNs
numeric_cols = ['AB', 'H', 'HR', 'SB', 'G', 'OBP', 'SLG', 'WAR', 'Age']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# Filter hitters
data = data[data['PA'] > 0].sort_values(['Player', 'Year'])

# Feature engineering
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

# Verify 'Player' column exists
assert 'Player' in data.columns, "Column 'Player' is missing!"

# Apply feature engineering
data = data.groupby('Player', group_keys=False).apply(create_features)

# Create target variable (Next_WAR)
data['Next_WAR'] = data.groupby('Player')['WAR'].shift(-1)
data = data.dropna(subset=['Next_WAR'])

# Define final features
features = [
    'Age', 'Career_G', 'Career_HR', 'Career_SB',
    'OBP_3yr', 'SLG_3yr', 'WAR', 'Post_Injury',
    'Peak_WAR', 'WAR_percentile'
]
target = 'Next_WAR'

# Train-test split
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

# Save model and scaler
joblib.dump(model, "war_predictor_final.pkl")
joblib.dump(scaler, "scaler_war_final.pkl")

print("Model trained successfully!")
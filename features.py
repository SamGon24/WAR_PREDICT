"""
Shared feature engineering for WAR prediction.

Imported by APP_FINAL.py, alt_predict_2.py, and alt_predict_graph.py
so that the feature matrix produced at inference time is identical to
what the model was trained on.
"""
import pandas as pd

# Ordered feature list expected by the model
FEATURES = [
    'Age', 'Career_G', 'Career_HR', 'Career_SB',
    'OBP_3yr', 'SLG_3yr', 'OPS_plus_3yr', 'WAR', 'WAR_trend', 'Peak_WAR',
    'Is_Rookie', 'Injured_Season', 'BB_rate', 'K_rate',
    'Age_from_peak', 'ISO', 'SB_pct',
]


def prepare_prediction_data(df, league_avgs):
    """Build the per-player feature row used for 2025 WAR inference.

    Parameters
    ----------
    df : pd.DataFrame
        Multi-year batting dataset (one row per player-season).
        Required columns: Player, Year, G, PA, OBP, SLG, WAR, HR, SB, Age.
    league_avgs : dict
        {'league_avg_obp': float, 'league_avg_slg': float}
        Used to impute OBP/SLG for first-year players.

    Returns
    -------
    pd.DataFrame
        One row per player with columns ['Player'] + FEATURES.
    """
    df = df.copy()

    # Career counting stats
    career_stats = df.groupby('Player').agg(
        Career_G=('G', 'sum'),
        Career_HR=('HR', 'sum'),
        Career_SB=('SB', 'sum'),
    ).reset_index()

    # Flag injured seasons so they are excluded from rolling averages
    df['Injured_Season'] = ((df['G'] < 50) | (df['PA'] < 200)).astype(int)

    # 3-year rolling OBP/SLG across healthy seasons only
    three_year_avg = (
        df.groupby('Player')
        .apply(lambda x: x[~x['Injured_Season'].astype(bool)].tail(3)[['OBP', 'SLG']].mean())
        .reset_index()
    )
    three_year_avg.columns = ['Player', 'OBP_3yr', 'SLG_3yr']

    # Most recent season row for each player
    latest = df.sort_values(['Player', 'Year']).groupby('Player').last().reset_index()

    # For players whose 2024 season was injury-shortened, fall back to 2023 stats
    injured_in_2024 = latest[
        (latest['Year'] == 2024) & ((latest['G'] < 50) | (latest['PA'] < 200))
    ]['Player']
    fallback = (
        df[df['Player'].isin(injured_in_2024) & (df['Year'] == 2023)]
        .set_index('Player')
    )
    latest.loc[latest['Player'].isin(injured_in_2024)] = (
        fallback.reindex(latest['Player'].values).reset_index()
    )

    # Merge aggregated features
    latest = latest.merge(career_stats, on='Player').merge(three_year_avg, on='Player')

    # Age projection: predicting the next season
    latest['Age'] = latest['Age'] + 1

    # Non-linear age curve: distance from peak age (27), squared
    latest['Age_from_peak'] = (latest['Age'] - 27) ** 2

    # Isolated power: strips singles out of SLG
    latest['ISO'] = latest['SLG'] - latest['BA']

    # Stolen base efficiency (0 if no attempts)
    latest['SB_pct'] = latest['SB'] / (latest['SB'] + latest['CS']).replace(0, float('nan'))
    latest['SB_pct'] = latest['SB_pct'].fillna(0)

    # Rookie flag: player has only one season of MLB data
    latest['Is_Rookie'] = (latest['Career_G'] == latest['G']).astype(int)

    # Plate discipline rates
    latest['BB_rate'] = latest['BB'] / latest['PA']
    latest['K_rate'] = latest['SO'] / latest['PA']

    # 3-year rolling OPS+ (park- and league-adjusted), excluding injured seasons
    df['OPS_plus_healthy'] = df['OPS+'].where(df['Injured_Season'] == 0)
    ops_plus_3yr = (
        df.groupby('Player')['OPS_plus_healthy']
        .apply(lambda x: x.rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    ops_plus_3yr = ops_plus_3yr.groupby(df['Player']).transform(lambda x: x.fillna(x.mean()))
    latest['OPS_plus_3yr'] = (
        df.assign(OPS_plus_3yr=ops_plus_3yr)
        .sort_values(['Player', 'Year'])
        .groupby('Player')['OPS_plus_3yr']
        .last()
        .reindex(latest['Player'].values)
        .values
    )

    # WAR trajectory: change vs two seasons ago (0 if insufficient history)
    war_sorted = df.sort_values(['Player', 'Year']).groupby('Player')['WAR']
    war_2yr_ago = war_sorted.apply(
        lambda x: x.iloc[-3] if len(x) >= 3 else x.iloc[0]
    )
    latest['WAR_trend'] = (
        war_sorted.last() - war_2yr_ago
    ).fillna(0).reindex(latest['Player'].values).values

    # Career peak WAR
    latest['Peak_WAR'] = (
        df.groupby('Player')['WAR'].max()
        .reindex(latest['Player'].values)
        .values
    )

    # Impute rookies with league-average rates and zero career counting stats
    rookie_mask = latest['Is_Rookie'] == 1
    latest.loc[rookie_mask, 'OBP_3yr'] = league_avgs['league_avg_obp']
    latest.loc[rookie_mask, 'SLG_3yr'] = league_avgs['league_avg_slg']
    latest.loc[rookie_mask, 'BB_rate'] = league_avgs['league_avg_bb_rate']
    latest.loc[rookie_mask, 'K_rate'] = league_avgs['league_avg_k_rate']
    latest.loc[rookie_mask, 'WAR_trend'] = 0
    latest.loc[rookie_mask, ['Career_G', 'Career_HR', 'Career_SB']] = 0

    # Current-season injury flag (used as a model feature)
    latest['Injured_Season'] = ((latest['G'] < 50) | (latest['PA'] < 200)).astype(int)

    return latest[['Player'] + FEATURES]

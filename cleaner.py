import pandas as pd

def clean_dataset(input_file, output_file):
    """
    Cleans a baseball dataset by:
    1. Removing players who didn't play in 2024
    2. Keeping all seasons' data for players who did play in 2024
    3. Ensuring consistent formatting
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV file
    """
    
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Standardize column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # Convert Year column to numeric (if not already)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Identify players who played in 2024 (appeared in games or had plate appearances)
    players_2024 = df[
        (df['Year'] == 2024) & 
        ((df['G'] > 0) | (df['PA'] > 0))
    ]['Player'].unique()
    
    # Filter original dataset to only include players who played in 2024
    cleaned_df = df[df['Player'].isin(players_2024)]
    
    # Additional cleaning steps
    cleaned_df = cleaned_df.dropna(subset=['Player'])  # Remove rows with missing player names
    cleaned_df = cleaned_df.sort_values(['Player', 'Year'])  # Sort by player and year
    
    # Save cleaned dataset
    cleaned_df.to_csv(output_file, index=False)
    print(f"Dataset cleaned. Saved to {output_file}")
    print(f"Original players: {len(df['Player'].unique())}")
    print(f"Players remaining: {len(cleaned_df['Player'].unique())}")

# Example usage
clean_dataset("sorted_dataset.csv", "cleaned_dataset_2024_players.csv")
import pandas as pd

def filter_players_2023(file_path, output_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Identify players who played in 2023
    players_2023 = set(df[df['Year'] == 2023]['Player'])
    
    # Filter the dataset to keep only relevant players and remove 2024
    df_filtered = df[df['Player'].isin(players_2023) & (df['Year'] <= 2023)]
    
    # Save the filtered dataset
    df_filtered.to_csv(output_path, index=False)
    
    print(f"Filtered dataset saved to {output_path}")

# Example usage
filter_players_2023('sorted_dataset.csv', 'filtered_stats.csv')

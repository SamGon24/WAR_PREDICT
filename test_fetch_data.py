from pybaseball import batting_stats
import pandas as pd

# Define the seasons to fetch
start_year = 2015
end_year = 2023

# Initialize an empty DataFrame to store all data
all_data = pd.DataFrame()

print("Fetching MLB batting stats from 2015 to 2023...\n")

# Loop through each season and fetch data
for year in range(start_year, end_year + 1):
    print(f"Fetching {year} data...")
    season_data = batting_stats(year)
    season_data["Season"] = year  # Add a season column for reference
    all_data = pd.concat([all_data, season_data], ignore_index=True)

# Save the compiled data to CSV
csv_filename = "mlb_batting_stats_2015_2023.csv"
all_data.to_csv(csv_filename, index=False)

print(f"\nData successfully saved to {csv_filename} with {len(all_data)} records.")

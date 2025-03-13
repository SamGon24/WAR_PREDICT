from pybaseball import batting_stats

# Fetch MLB batting stats for 2023 season (without filtering by missing data)
data = batting_stats(2023)

# Display the first few rows to confirm it's pulling more players
print(data.head())

# Save the data to a CSV file
data.to_csv('mlb_hitters_2023.csv', index=False)

# Check the number of rows to confirm the number of players fetched
print(f"Total number of players fetched: {len(data)}")

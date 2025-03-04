from pybaseball import batting_stats
import pandas as pd


# Fetch MLB batting stats for the 2023 season
data = batting_stats(2023)

# Display the first 5 rows
print(data.head())
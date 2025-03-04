from pybaseball import batting_stats
import pandas as pd

# Fetch batting stats from 2010 to 2023
df = batting_stats(2010, 2023)

# Display the first few rows
print(df.head())

import pandas as pd

# Load your dataset (update the filename if needed)
df = pd.read_csv("sorted_dataset.csv")

# Check for duplicate players in the same year
duplicates = df[df.duplicated(subset=["Player", "Year"], keep=False)]

# Display the duplicates if found
if not duplicates.empty:
    print("Duplicate players found in the same year:")
    print(duplicates.sort_values(by=["Year", "Player"]))
else:
    print("No duplicate players found in the same year.")

# Optionally, save the duplicates to a CSV file for review
duplicates.to_csv("duplicate_players.csv", index=False)
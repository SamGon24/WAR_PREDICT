import pandas as pd

# Load the dataset
data = pd.read_csv("new_data - 2015_updated.csv")

# Keep the row with the highest AB for each player
data = data.sort_values(by="AB", ascending=False).drop_duplicates(subset="Player", keep="first")

# Save the cleaned dataset
data.to_csv("new_data - 2015_updated.csv", index=False)

print("Cleaned dataset saved as 'cleaned_mlb_data.csv'.")



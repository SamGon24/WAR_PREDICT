import pandas as pd

# Load the dataset
data = pd.read_csv("new_data - 2015_updated.csv")

# Remove pitchers (players with AB <= 0)
data = data[data["AB"] > 0]

# Save the cleaned dataset
data.to_csv("new_data - 2015_updated.csv", index=False)

print("Cleaned dataset saved as 'cleaned_hitters_data.csv'.")

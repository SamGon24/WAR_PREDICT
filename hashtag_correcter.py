import pandas as pd

# Load the dataset
data = pd.read_csv("new_data - 2022_updated.csv")

# Remove asterisks (*) and hashtags (#) from the 'Name' column
data["Player"] = data["Player"].str.replace(r"[*#]", "", regex=True)

# Save the cleaned dataset (optional)
data.to_csv("new_data - 2022_updated.csv", index=False)
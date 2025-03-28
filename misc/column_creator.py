import pandas as pd

# Load the CSV file
df = pd.read_csv("standard_fielding.csv")

# Insert a new column 'Year' at the first position
df.insert(0, "Year", "2024")

# Save the updated CSV
df.to_csv("standard_fielding.csv", index=False)

print("New column 'Year' added successfully!")

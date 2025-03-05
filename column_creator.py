import pandas as pd

# Load the CSV file
df = pd.read_csv("new_data - 2022_updated.csv")

# Insert a new column 'Year' at the first position
df.insert(0, "Year", "2022")

# Save the updated CSV
df.to_csv("new_data - 2022_updated.csv", index=False)

print("New column 'Year' added successfully!")

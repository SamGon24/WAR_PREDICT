import pandas as pd

# Load the dataset
df = pd.read_csv("cleaned_dataset.csv")

# Ensure 'Year' and 'Rk' columns are of the correct type
df['Year'] = df['Year'].astype(int)
df['Rk'] = df['Rk'].astype(int)

# Sort first by Year (descending), then by Rk (descending)
df_sorted = df.sort_values(by=['Year', 'Rk'], ascending=[False, True])

# Save the cleaned dataset
df_sorted.to_csv("sorted_dataset.csv", index=False)

print("Dataset sorted successfully. Saved as 'sorted_dataset.csv'.")

import pandas as pd

# Load your dataset
file_path = "merged_data_2015_2024.csv"  # Change this to your actual file
df = pd.read_csv(file_path)

# Filter only the 2023 season
df_2023 = df[df['Year'] == 2023]

# Keep only the row with the highest AB per player
df_2023_cleaned = df_2023.loc[df_2023.groupby('Player')['AB'].idxmax()]

# Remove old 2023 data from the main dataset and add the cleaned 2023 data back
df_cleaned = df[df['Year'] != 2023]  # Remove all 2023 data
df_cleaned = pd.concat([df_cleaned, df_2023_cleaned], ignore_index=True)  # Add cleaned 2023 data

# Save the cleaned dataset
df_cleaned.to_csv("cleaned_dataset.csv", index=False)

print("Dataset cleaned! Removed duplicate 2023 player rows, keeping highest AB.")

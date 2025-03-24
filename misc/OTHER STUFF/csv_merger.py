import pandas as pd
import glob

# Get all CSV files in the current directory (change path if needed)
csv_files = ["new_data - 2024_updated.csv", "new_data - 2023_updated.csv", "new_data - 2022_updated.csv", 
             "new_data - 2021_updated.csv", "new_data - 2019_updated.csv", "new_data - 2018_updated.csv",
             "new_data - 2017_updated.csv", "new_data - 2016_updated.csv", "new_data - 2015_updated.csv"]  # Manually list or use glob to fetch files

# Read and concatenate them
df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)  # Combine and reset index

# Save the merged dataset
merged_df.to_csv("merged_data_2015_2024.csv", index=False)

print("Merged dataset saved as 'merged_data.csv'.")

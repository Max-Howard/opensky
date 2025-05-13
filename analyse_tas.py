import pandas as pd
import glob
import os

import matplotlib.pyplot as plt

# Function to read CSV files and calculate the difference between TAS and GS
def read_and_calculate_diff(file_path):
    df = pd.read_csv(file_path)
    df['Tailwind'] = df['gs'] - df['tas']
    return df

# Directory containing the CSV files
csv_dir = './ProcessedFlightData/'

# Read all CSV files in the directory
all_files = glob.glob(os.path.join(csv_dir, "*.csv"))
print(f"Found {len(all_files)} CSV files.")

# Concatenate all dataframes
df_list = [read_and_calculate_diff(file) for file in all_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Group the differences into 1 m/s bins
combined_df['Tailwind'] = combined_df['Tailwind'].round()

# Count the number of points in each group
grouped_df = combined_df.groupby('Tailwind').size().reset_index(name='Count')

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(grouped_df['Tailwind'], grouped_df['Count'], width=0.8, color='blue', alpha=0.7)
plt.xlabel('Tailwind (m/s)')
plt.ylabel('Number of Points')
plt.title('Variation of TAS and GS Difference')
plt.grid(True)
plt.show()
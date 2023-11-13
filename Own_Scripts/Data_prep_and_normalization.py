import pandas as pd
import glob
import os
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Directory where the CSV files are stored
input_folder_countries = r"C:\Backup-Folder\Masterarbeit_Klemens Lechner\Thesis_Modell_Training\Data\Raw\Excel_by_countries"
input_folder_gas = r"C:\Backup-Folder\Masterarbeit_Klemens Lechner\Thesis_Modell_Training\Data\Raw\Gas_price\Gas_prices.xlsx"

# Define the output directory for Excel files
output_folder = r"C:\Backup-Folder\Masterarbeit_Klemens Lechner\Thesis_Modell_Training\Data\Processed_data"

#Load gas prices
gas_prices = pd.read_excel(input_folder_gas)
gas_prices['Timestamp'] = pd.to_datetime(gas_prices['Timestamp'])
gas_prices['Timestamp'] = gas_prices['Timestamp'].dt.tz_localize(None)
gas_prices.sort_values('Timestamp', inplace=True)

#Calculate moving average

gas_prices['Moving_Average'] = gas_prices['gas_price'].rolling(window = 24*365).mean()

# Find all CSV files in the directory
files = glob.glob(f"{input_folder_countries}\\*.xlsx")

# Dictionary to organize DataFrames by country
dfs_by_country = defaultdict(list)

# Read each file, load it into a DataFrame, and organize by country code
for file in files:
    country_code = os.path.basename(file).split("_")[0]
    df = pd.read_excel(file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
    df.sort_values('Timestamp', inplace=True)
    dfs_by_country[country_code].append(df)

print('Data has been loaded')

# Get all columns
all_columns = set()
for df_list in dfs_by_country.values():
    for df in df_list:
        all_columns.update(df.columns)

# Add missing columns and set them to be zero and sort dataframe
for country_code, df_list in dfs_by_country.items():
    for i, df in enumerate(df_list):
        missing_columns = all_columns.difference(df.columns)
        for column in missing_columns:
            df[column] = 0
        df = df.reindex(columns=sorted(all_columns))
        df_list[i] = df

# Merge each country DataFrame with the gas prices DataFrame
for country_code, df_list in dfs_by_country.items():
    for i in range(len(df_list)):
        df = df_list[i]
        merged_df = pd.merge_asof(df, gas_prices.sort_values('Timestamp'), on="Timestamp")
        dfs_by_country[country_code][0] = merged_df

print('Gas prices and Moving Average added')

# Drop Day_Ahead_Total_Load
for country_code, df_list in dfs_by_country.items():
    for i in range(len(df_list)):
        df = df_list[i]
        df.drop(['Day_Ahead_Total_Load'], axis=1, inplace=True)
        dfs_by_country[country_code][0] = df

print('Day_Ahead_Total_Load dropped')

#Calculate "Total Generation"
for country_code, df_list in dfs_by_country.items():
    for i in range(len(df_list)):
        df = df_list[i]
        columns_to_exclude = ["Timestamp", "Actual_Total_Load", "Day_Ahead_Prices", "gas_price", "Moving_Average"]
        columns_to_sum = df.columns.difference(columns_to_exclude)
        dfs_by_country[country_code][0] = df
        df['Total_Generation'] = df[columns_to_sum].sum(axis=1)
        output_file_path = os.path.join(output_folder, f"{country_code}_combined.xlsx")
        df.to_excel(output_file_path, index=False)

print('Total Generation added')


#Get the relative Generation from each power source
for country_code, df_list in dfs_by_country.items():
    for i in range(len(df_list)):
        df = df_list[i]
        columns_to_exclude = ["Timestamp", "Actual_Total_Load", "Day_Ahead_Prices", "gas_price", "Total_Generation", "Moving_Average"]
        columns_to_divide = df.columns.difference(columns_to_exclude)
        for column in columns_to_divide:
            df[column] = df[column] / df['Total_Generation']
        output_file_path = os.path.join(output_folder, f"{country_code}_combined.xlsx")
        df.to_excel(output_file_path, index=False)

print('Generation per type has been normalized')

#Include "Day_of_the_year"
for country_code, df_list in dfs_by_country.items():
    for i in range(len(df_list)):
        df = df_list[i]
        df['Day_of_the_year'] = df['Timestamp'].dt.dayofyear

print('Day of the year added')

# Normalize Actual total Load with minmax-Scaler
scaler = MinMaxScaler()
for country_code, df_list in dfs_by_country.items():
    for i in range(len(df_list)):
        df = df_list[i]
        df['Actual_Total_Load'] = scaler.fit_transform(df[['Actual_Total_Load']])
        df['Total_Generation'] = scaler.fit_transform(df[['Total_Generation']])
        df['gas_price'] = scaler.fit_transform(df[['gas_price']])
        df['Moving_Average'] = scaler.fit_transform(df[['Moving_Average']])
        dfs_by_country[country_code][0] = df

print('Normalized')

# Initialisiere Variablen für die maximalen und minimalen Werte
max_day_ahead_price = float('-inf')  # Setze den Anfangswert auf minus unendlich
min_day_ahead_price = float('inf')  # Setze den Anfangswert auf unendlich

#Find min and max of all "Day ahead Prices" for minmax scaling
for country_code, dfs in dfs_by_country.items():
    for df in dfs:
        max_price = df['Day_Ahead_Prices'].max()
        min_price = df['Day_Ahead_Prices'].min()

        if max_price > max_day_ahead_price:
            max_day_ahead_price = max_price
            print(f"Found new max: {max_day_ahead_price} for {country_code}")
        if min_price < min_day_ahead_price:
            min_day_ahead_price = min_price
            print(f"Found new min: {min_day_ahead_price} for {country_code}")

for country_code, df_list in dfs_by_country.items():
    for i in range(len(df_list)):
        df = df_list[i]
        df['Day_Ahead_Prices'] = (df['Day_Ahead_Prices'] - min_day_ahead_price) / (max_day_ahead_price - min_day_ahead_price)
        dfs_by_country[country_code][0] = df
        output_file_path = os.path.join(output_folder, f"{country_code}_combined.xlsx")
        df.to_excel(output_file_path, index=False)

print('Day ahead prices normalized')

# Day of the year scaling (sin because it is a cycle)
for country_code, dfs in dfs_by_country.items():
    for df in dfs:
        df['Day_of_Year_sin'] = np.sin(2 * np.pi * df['Day_of_the_year'] / 365.25)
        df['Day_of_Year_cos'] = np.cos(2 * np.pi * df['Day_of_the_year'] / 365.25)
        df.drop(['Day_of_the_year'], axis=1, inplace=True)
        output_file_path = os.path.join(output_folder, f"{country_code}_combined.xlsx")
        df.to_excel(output_file_path, index=False)


print('Day of the year normalized')

# Save all dataframes as pickle
pickle_file_path = os.path.join(output_folder, 'All_dataframes.pkl')

pd.to_pickle(dfs_by_country, pickle_file_path)









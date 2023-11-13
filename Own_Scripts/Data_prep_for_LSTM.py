import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from collections import defaultdict

input_path_pickle = r"C:\Backup-Folder\Masterarbeit_Klemens Lechner\Thesis_Modell_Training\Data\Processed_data\All_dataframes.pkl"

dfs_by_country = pd.read_pickle(input_path_pickle)

features = defaultdict(list)
labels = defaultdict(list)

# Drop Timestamp
for country_code, df_list in dfs_by_country.items():
    for i in range(len(df_list)):
        df = df_list[i]
        df.drop(['Timestamp'], axis=1, inplace=True)
        dfs_by_country[country_code][i] = df

#Split features and labels
for country_code, dfs in dfs_by_country.items():
    for df in dfs:
        df_features = df.drop('Day_Ahead_Prices', axis=1)
        df_label = df['Day_Ahead_Prices']
        features[country_code].append(df_features)
        labels[country_code].append(df_label)

for country_code, dfs in features.items():
    for df_index, df in enumerate(dfs):
        # Identifiziere die Spalten mit NaN-Werten und deren Anzahl
        nan_counts = df.isnull().sum()
        nan_columns = nan_counts[nan_counts > 0].index.tolist()
        print(f"DataFrame {df_index} for Country Code {country_code} has NaN in columns: {nan_columns}")

        for column in nan_columns:
            print(f"Anzahl der NaN-Werte in der Spalte '{column}': {nan_counts[column]}")

for country_code, dfs in features.items():
    for df_index, df in enumerate(dfs):
        # Identifiziere die Spalten mit unendlichen Werten und deren Anzahl
        inf_counts = np.isinf(df).sum()

        inf_columns = inf_counts[inf_counts > 0].index.tolist()

        print(f"DataFrame {df_index} for Country Code {country_code} has inf in following columns: {inf_columns}")

        for column in inf_columns:
            print(f"Anzahl der unendlichen Werte in der Spalte '{column}': {inf_counts[column]}")





#Create the sequences
Xs, ys = [], []
sequence_length=24
for country_code, dfs in features.items():
    for df in dfs:
        for i in range(len(df) - sequence_length):
            Xs.append(df.iloc[i:(i + sequence_length)])

for country_code, dfs in labels.items():
    for df in dfs:
        for i in range(len(df) - sequence_length):
            ys.append(df.iloc[i + sequence_length])

# Shuffle the data
np.random.seed(42)
Xs, ys = shuffle(Xs, ys)

# Datasplit
n = len(Xs)
train_size = int(n * 0.7)
val_size = test_size = int((n - train_size) / 2)

X_train = Xs[:train_size]
y_train = ys[:train_size]

X_val = Xs[train_size:train_size+val_size]
y_val = ys[train_size:train_size+val_size]

X_test = Xs[train_size+val_size:]
y_test = ys[train_size+val_size:]

# Speichern der Sequenzen
np.savez(r'C:\Backup-Folder\Masterarbeit_Klemens Lechner\Thesis_Modell_Training\Data\Data for LSTM\v1\Training Data\train_data.npz', sequences=X_train, labels=y_train)
np.savez(r'C:\Backup-Folder\Masterarbeit_Klemens Lechner\Thesis_Modell_Training\Data\Data for LSTM\v1\Training Data\val_data.npz', sequences=X_val, labels=y_val)
np.savez(r'C:\Backup-Folder\Masterarbeit_Klemens Lechner\Thesis_Modell_Training\Data\Data for LSTM\v1\Training Data\test_data.npz', sequences=X_test, labels=y_test)





import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime

train_data_path = r'C:\Backup-Folder\Masterarbeit_Klemens Lechner\Lechner_thesis\Data\Data for LSTM\v1\Training Data\train_data.npz'
val_data_path = r'C:\Backup-Folder\Masterarbeit_Klemens Lechner\Lechner_thesis\Data\Data for LSTM\v1\Training Data\val_data.npz'
test_data_path = r'C:\Backup-Folder\Masterarbeit_Klemens Lechner\Lechner_thesis\Data\Data for LSTM\v1\Training Data\test_data.npz'

# Daten laden
train_data = np.load(train_data_path)
val_data = np.load(val_data_path)
test_data = np.load(test_data_path)

# Zuweisen der Sequenzen und Labels
sequences_train = train_data['sequences']
labels_train = train_data['labels']
sequences_val = val_data['sequences']
labels_val = val_data['labels']
sequences_test = test_data['sequences']
labels_test = test_data['labels']

# Zählen der NaN-Werte in den Trainingssequenzen
nan_count_sequences = np.isnan(sequences_train).sum()
inf_count_sequences = np.isinf(sequences_train).sum()

# Zählen der NaN-Werte in den Trainingslabels
nan_count_labels = np.isnan(labels_train).sum()
inf_count_labels = np.isinf(labels_train).sum()

print(f"Anzahl der NaN-Werte in Sequences: {nan_count_sequences}")
print(f"Anzahl der Inf-Werte in Sequences: {inf_count_sequences}")
print(f"Anzahl der NaN-Werte in Labels: {nan_count_labels}")
print(f"Anzahl der Inf-Werte in Labels: {inf_count_labels}")



# Modell-Parameter
input_shape = (sequences_train.shape[1], sequences_train.shape[2])
output_size = 1

# Modell erstellen
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=input_shape))
model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
model.add(LSTM(64))
model.add(Dense(output_size))

# Modell kompilieren
model.compile(optimizer='adam', loss='mean_squared_error')

# Callbacks für das Training
model_checkpoint = ModelCheckpoint(r'C:\Backup-Folder\Masterarbeit_Klemens Lechner\Lechner_thesis\Data\Data for LSTM\v1\Training Data\best_model_v1.h5', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Training des Modells
history = model.fit(sequences_train, labels_train,
                    epochs=100,
                    batch_size=1,
                    validation_data=(sequences_val, labels_val),
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1)
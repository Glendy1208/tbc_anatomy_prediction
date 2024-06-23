import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Langkah 1: Mengambil data dari file CSV
df = pd.read_csv('tbc_clean.csv')

# Langkah 2: Membagi data menjadi fitur dan label
X = df.drop(columns=['lokasi_anatomi'])
y = df['lokasi_anatomi']

# Langkah 3: Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Langkah tambahan: Standardisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Simpan scaler
import joblib
joblib.dump(scaler, 'scaler.joblib')

# Langkah 4: Membangun model neural network dengan 6 hidden layer
model = Sequential()

# input layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# hidden
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))

# output layer
model.add(Dense(1, activation='sigmoid'))  # klasifikasi biner

# Langkah 5: Melatih model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Simpan model
model.save('tbc_model.h5')
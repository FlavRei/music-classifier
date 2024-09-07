import os
import pickle
import numpy as np
from src.utils import save_model
from src.model_training import build_cnn
from src.data_processing import load_audio_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

data_path = './data/raw'
genres = os.listdir(data_path)
X, y = load_audio_files(data_path, genres)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

importances = RandomForestClassifier().fit(X_scaled, y_categorical).feature_importances_
indices = np.argsort(importances)[::-1]
importance_threshold = 0.015
important_features_indices = [i for i, importance in enumerate(importances) if importance >= importance_threshold]
X_filtered = X_scaled[:, important_features_indices]

X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_categorical, test_size=0.2, random_state=42)

X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

cnn_model = build_cnn((X_train_reshaped.shape[1], X_train_reshaped.shape[2]), y_train.shape[1])
cnn_model.fit(X_train_reshaped, y_train, epochs=150, batch_size=32, validation_data=(X_test_reshaped, y_test))

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

np.save('models/important_features.npy', important_features_indices)

save_model(cnn_model, 'models/cnn_model.h5')
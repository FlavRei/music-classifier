import numpy as np
from src.data_processing import extract_instrument_features

def predict_genre(model, file_path, scaler, label_encoder, important_features_indices):
    features = extract_instrument_features(file_path)
    if features.ndim == 1:
        features = features.reshape(1, -1)
        
    features_scaled = scaler.transform(features)
    features_filtered = features_scaled[:, important_features_indices]
    features_reshaped = features_filtered.reshape((features_filtered.shape[0], features_filtered.shape[1], 1))
    predictions = model.predict(features_reshaped)
    predicted_class = np.argmax(predictions)
    predicted_genre = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_genre

import os
import librosa
import numpy as np

def load_audio_files(data_path, genres):
    all_features = []
    all_labels = []
    for genre in genres:
        for file in os.listdir(os.path.join(data_path, genre)):
            file_path = os.path.join(data_path, genre, file)
            features = extract_instrument_features(file_path)
            all_features.append(features)
            all_labels.append(genre)
    return np.array(all_features), np.array(all_labels)

def extract_instrument_features(file_path):
    y, sr = librosa.load(file_path, duration=240)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features = np.hstack([np.mean(mfccs.T, axis=0), np.mean(chroma.T, axis=0),
                          np.mean(spectral_contrast.T, axis=0), np.mean(tonnetz.T, axis=0),
                          np.mean(zero_crossing_rate.T, axis=0), np.mean(spectral_centroid.T, axis=0),
                          np.mean(spectral_bandwidth.T, axis=0), np.mean(spectral_rolloff.T, axis=0)])
    return features

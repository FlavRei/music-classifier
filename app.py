import pickle
import numpy as np
import streamlit as st
from src.utils import load_model
from src.predict_genre import predict_genre

MODEL_PATH = './models/cnn_model.h5'
SCALER_PATH = './models/scaler.pkl'
LABEL_ENCODER_PATH = './models/label_encoder.pkl'
IMPORTANT_FEATURES_PATH = './models/important_features.npy'

model = load_model(MODEL_PATH)
scaler = pickle.load(open(SCALER_PATH, 'rb'))
label_encoder = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))
important_features_indices = np.load(IMPORTANT_FEATURES_PATH)

st.title("Music classifier")

audio_file = st.file_uploader("Upload an audio file to predict its genre.", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    
    genre = predict_genre(model, "temp_audio.wav", scaler, label_encoder, important_features_indices)
    st.markdown(f"<p style='font-size: 24px;'>The genre of this music is <strong>{genre}</strong></p>", unsafe_allow_html=True)

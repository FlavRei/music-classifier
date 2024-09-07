from tensorflow.keras.models import load_model as keras_load_model

def save_model(model, filename):
    model.save(filename)

def load_model(filename):
    return keras_load_model(filename)

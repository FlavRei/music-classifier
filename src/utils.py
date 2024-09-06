import joblib

def save_model(model, filename):
    model.save(filename)

def load_model(filename):
    return joblib.load(filename)

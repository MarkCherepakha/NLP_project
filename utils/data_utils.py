import os
import joblib
from utils.constants import PREPROCESSED_DATA_FILE_PATH, MODEL_FILE_PATH

def save_preprocessed_data(data):
    joblib.dump(data, PREPROCESSED_DATA_FILE_PATH)
    print(f'Data saved to {PREPROCESSED_DATA_FILE_PATH}')

def load_preprocessed_data():
    if os.path.exists(PREPROCESSED_DATA_FILE_PATH):
        data = joblib.load(PREPROCESSED_DATA_FILE_PATH)
        print(f'Data loaded from {PREPROCESSED_DATA_FILE_PATH}')
        return data
    else:
        print(f'No file found at {PREPROCESSED_DATA_FILE_PATH}')
        return None

def save_model(model):
    joblib.dump(model, MODEL_FILE_PATH)
    print(f'Model saved to {MODEL_FILE_PATH}')

def load_model():
    if os.path.exists(MODEL_FILE_PATH):
        model = joblib.load(MODEL_FILE_PATH)
        print(f'Model loaded from {MODEL_FILE_PATH}')
        return model
    else:
        print(f'No file found at {MODEL_FILE_PATH}')
        return None

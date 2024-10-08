import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MODEL_FILE_PATH = os.path.join(MODELS_DIR, 'model.joblib')
PREPROCESSED_DATA_FILE_PATH = os.path.join(DATA_DIR, 'preprocessed_data.joblib')

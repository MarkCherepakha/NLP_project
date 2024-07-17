import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

MODEL_FILENAME = 'model.joblib'
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)

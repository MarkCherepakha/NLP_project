import joblib
import os
from utils.constants import DATA_DIR

def save_preprocessed_data(data, filename='preprocessed_data.joblib'):
    filepath = os.path.join(DATA_DIR, filename)
    joblib.dump(data, filepath)
    print(f'Data saved to {filepath}')

preprocessed_data = ["example", "preprocessed", "data"]
save_preprocessed_data(preprocessed_data)

def load_preprocessed_data(filename='preprocessed_data.joblib'):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        data = joblib.load(filepath)
        print(f'Data loaded from {filepath}')
        return data
    else:
        print(f'No file found at {filepath}')
        return None

loaded_data = load_preprocessed_data()
print(loaded_data)

from utils.data_utils import save_preprocessed_data, load_preprocessed_data, save_model, load_model
from utils.src.nlp_utils.preprocessing.text_preprocessing import text_preprocessing

data = ["This is an example sentence.", "This is another sentence."]

preprocessed_data = [text_preprocessing(text) for text in data]

save_preprocessed_data(preprocessed_data)

model = load_model()

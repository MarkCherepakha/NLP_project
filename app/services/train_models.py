from utils.data_utils import save_preprocessed_data, load_preprocessed_data, save_model, load_model
from utils.src.nlp_utils.preprocessing.text_preprocessing import text_preprocessing

data = ["Inulinases are used for the production of high-fructose syrup and fructooligosaccharides, and are widely utilized in food and pharmaceutical industries.", "In this study, different carbon sources were screened for inulinase production by Aspergillus niger in shake flask fermentation."]

preprocessed_data = [text_preprocessing(text) for text in data]

save_preprocessed_data(preprocessed_data)

model = load_model()

import re
import nltk
import spacy
import pandas as pd
import joblib
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils.nltk_resources import download_nltk_resources


download_nltk_resources()

nlp = spacy.load('en_core_web_sm')

def text_preprocessing(text):
    standardized_text = text.lower()
    punctuated_text = re.sub(r'[^\w\s]', '', standardized_text)
    text_numbers_removed = re.sub(r'\d', '', punctuated_text)

    def remove_rare_words(text, threshold=5):
        words = nltk.word_tokenize(text)
        word_freq = Counter(words)
        filtered_words = [word for word in words if word_freq[word] >= threshold]
        return ' '.join(filtered_words)

    text_removed_from_rare_words = remove_rare_words(text_numbers_removed, threshold=2)

    tokens = word_tokenize(text_removed_from_rare_words)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    sentence = " ".join(filtered_tokens)
    lemmatized_tokens = [nlp(token)[0].lemma_ for token in filtered_tokens]

    pos_tag_tokens = word_tokenize(text_removed_from_rare_words)
    tags = nltk.pos_tag(pos_tag_tokens)
    chunks = nltk.ne_chunk(tags)
    df = pd.DataFrame(chunks, columns=['Word', 'IOB Notation'])

    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text_removed_from_rare_words)

    return {
        'standardized_text': standardized_text,
        'punctuated_text': punctuated_text,
        'text_numbers_removed': text_numbers_removed,
        'text_removed_from_rare_words': text_removed_from_rare_words,
        'tokens': tokens,
        'filtered_tokens': filtered_tokens,
        'stemmed_tokens': stemmed_tokens,
        'lemmatized_tokens': lemmatized_tokens,
        'pos_tags': df,
        'sentiment_scores': scores
    }

def spacy_preprocessing(text):
    doc = nlp(text)

    standardized_text = text.lower()

    punctuated_text = re.sub(r'[^\w\s]', '', standardized_text)
    text_numbers_removed = re.sub(r'\d', '', punctuated_text)

    words = [token.text for token in doc if token.is_alpha]
    word_freq = Counter(words)
    filtered_words = [word for word in words if word_freq[word] >= 2]
    text_removed_from_rare_words = ' '.join(filtered_words)

    tokens = [token.text for token in doc]
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    lemmatized_tokens = [token.lemma_ for token in doc]
    pos_tags = [(token.text, token.pos_) for token in doc]

    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text_removed_from_rare_words)

    return {
        'tokens': tokens,
        'filtered_tokens': filtered_tokens,
        'lemmatized_tokens': lemmatized_tokens,
        'pos_tags': pos_tags,
        'sentiment_scores': scores
    }

text = """Inulinases are used for the production of high-fructose syrup and fructooligosaccharides, and are widely utilized in food and pharmaceutical industries. In this study, different carbon sources were screened for inulinase production by Aspergillus niger in shake flask fermentation. Optimum working conditions of the enzyme were determined. Additionally, some properties of produced enzyme were determined [activation (Ea)/inactivation (Eia) energies, Q10 value, inactivation rate constant (kd), half-life (t1/2), D value, Z value, enthalpy (ΔH), free energy (ΔG), and entropy (ΔS)]. Results showed that sugar beet molasses (SBM) was the best in the production of inulinase, which gave 383.73 U/mL activity at 30 °C, 200 rpm and initial pH 5.0 for 10 days with 2% (v/v) of the prepared spore solution. Optimum working conditions were 4.8 pH, 60 °C, and 10 min, which yielded 604.23 U/mL, 1.09 inulinase/sucrase ratio, and 2924.39 U/mg. Additionally, Ea and Eia of inulinase reaction were 37.30 and 112.86 kJ/mol, respectively. Beyond 60 °C, Q10 values of inulinase dropped below one. At 70 and 80 °C, t1/2 of inulinase was 33.6 and 7.2 min; therefore, inulinase is unstable at high temperatures, respectively. Additionally, t1/2, D, ΔH, ΔG values of inulinase decreased with the increase in temperature. Z values of inulinase were 7.21 °C. Negative values of ΔS showed that enzymes underwent a significant process of aggregation during denaturation. Consequently, SBM is a promising carbon source for inulinase production by A. niger. Also, this is the first report on the determination of some properties of A. niger A42 (ATCC 204,447) inulinase."""

preprocessed_data = text_preprocessing(text)
for key, value in preprocessed_data.items():
    print(f"{key}: {value}\n")

print("WARNING: You shouldn't see this message if you try to import this module")

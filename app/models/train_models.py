import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

from preprocessing.text_preprocessing import text_preprocessing

def train():
    dataset = load_dataset('imdb')
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']

    train_texts = [text_preprocessing(text) for text in train_texts]
    test_texts = [text_preprocessing(text) for text in test_texts]

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(train_texts, train_labels)

    joblib.dump(model, 'model.joblib') # TODO save to model folder

    return model

if __name__ == "__main__":
    train()

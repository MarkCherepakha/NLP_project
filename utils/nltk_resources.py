import nltk
from nltk.data import find
from tqdm import tqdm


nltk_packages = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']

def download_nltk_packages(packages):
    for package in tqdm(packages, desc="Checking and downloading NLTK packages"):
        try:
            find(f'tokenizers/{package}')  # Проверяем наличие пакета
        except LookupError:
            nltk.download(package)  # Загружаем пакет, если он отсутствует

download_nltk_packages(nltk_packages)


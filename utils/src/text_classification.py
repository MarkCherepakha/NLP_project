from data.data_utils import load_model
from utils.src.nlp_utils.preprocessing.text_preprocessing import text_preprocessing

def classify_text(text):
    model = load_model()
    if model is None:
        return {"error": "Model not found"}
    
    preprocessed_text = text_preprocessing(text)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    from collections import Counter
    print(sorted(Counter(y).items()))
    print(sorted(Counter(y_resampled).items()))
    prediction = model.predict([preprocessed_text])
    
    return {"prediction": prediction}

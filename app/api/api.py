from typing import Any

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks

from models.model import PredictRequest, PredictResponse
from utils.src.nlp_utils.preprocessing.text_preprocessing import text_preprocessing

from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import textdistance
from typing import Dict

import joblib

model = joblib.load('model.joblib')
app = FastAPI()
api_router = APIRouter()

class TextClassificationRequest(BaseModel):
    text: str

class TextClassificationResponse(BaseModel):
    text: str
    label: int

@api_router.post("/classify", response_model=TextClassificationResponse)
def classify_text(request: TextClassificationRequest):
    try:
        preprocessed_text = text_preprocessing(request.text)
        prediction = model.predict([preprocessed_text])[0]
        return TextClassificationResponse(text=request.text, label=int(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/train")
def train_model(background_tasks: BackgroundTasks):
    def train():
        from datasets import load_dataset
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import make_pipeline

        from utils.src.nlp_utils.preprocessing.text_preprocessing import text_preprocessing

        dataset = load_dataset('imdb')
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        test_texts = dataset['test']['text']
        test_labels = dataset['test']['label']

        train_texts = [text_preprocessing(text) for text in train_texts]
        test_texts = [text_preprocessing(text) for text in test_texts]

        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(train_texts, train_labels)

        joblib.dump(model, 'model.joblib')

    background_tasks.add_task(train)
    return {"message": "Training started in the background"}

class SimilarityMethod(str, Enum):
    levenshtein = "levenshtein"
    jaccard = "jaccard"
    cosine = "cosine"

class SimilarityRequest(BaseModel):
    method: SimilarityMethod
    line1: str
    line2: str

@api_router.post("/similarity")
def calculate_similarity(request: SimilarityRequest) -> Dict[str, str]:
    method = request.method
    line1 = request.line1
    line2 = request.line2

    if method == SimilarityMethod.levenshtein:
        similarity = textdistance.levenshtein.normalized_similarity(line1, line2)
    elif method == SimilarityMethod.jaccard:
        similarity = textdistance.jaccard.normalized_similarity(line1, line2)
    elif method == SimilarityMethod.cosine:
        similarity = textdistance.cosine.normalized_similarity(line1, line2)
    else:
        raise HTTPException(status_code=400, detail="Invalid method")

    return {
        "method": method,
        "line1": line1,
        "line2": line2,
        "similarity": str(similarity)
    }


@api_router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest) -> Any:
    """
    ML Prediction API
    """
    input_text = payload.input_text
    model = request.app.state.model

    predict_value = model.predict(input_text)
    return PredictResponse(result=predict_value)

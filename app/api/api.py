from typing import Any

from fastapi import APIRouter, Request

from app.models.predict import PredictRequest, PredictResponse

from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import textdistance
from typing import Dict

class SimilarityMethod(str, Enum):
    levenshtein = "levenshtein"
    jaccard = "jaccard"
    cosine = "cosine"

class SimilarityRequest(BaseModel):
    method: SimilarityMethod
    line1: str
    line2: str

app = FastAPI()
api_router = APIRouter()

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
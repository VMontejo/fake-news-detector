from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import load_model
from .predict import predict_article

#initialize API
app = FastAPI(title = "Fake News Detector API")

#Define request model
class NewsArticle(BaseModel):
    title: str
    text: str

#Define response model
class PredictionResponse(BaseModel):
    label: int
    prediction: str
    confidence: float

#Load model and tokenizer
model, tokenizer = load_model()

@app.get("/")
async def root():
    return {"message": "Fake News Detector API is running"}

@app.get("/health")
async def health_check():
    return{"Status": "Healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(article: NewsArticle):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail= "Model not loaded")
    result = predict_article(model, tokenizer, article.title, article.text)
    return result

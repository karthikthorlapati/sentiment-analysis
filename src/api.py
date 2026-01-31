
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Twitter Sentiment Analysis API")

MODEL_PATH = os.getenv("MODEL_PATH", "model_output")
model = None
tokenizer = None

def clean_text(text):
    """
    Clean text by removing HTML tags, special characters, and converting to lowercase.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower().strip()
    return text

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    try:
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # We don't exit here so the container can still start and pass health check (maybe with status: model_not_loaded)
        # But for requirement 8, /health must return 200 OK. 
        # Requirement 9 /predict will fail if model is not loaded.

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text:
         raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure the model is trained and available in model_output/.")
        
    try:
        # Preprocess the text
        cleaned_text = clean_text(request.text)
        
        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        confidence, predicted_class = torch.max(probs, dim=1)
        # IMDb is binary: 0=negative, 1=positive
        labels = ["negative", "positive"] 
        sentiment = labels[predicted_class.item()]
        
        return {"sentiment": sentiment, "confidence": float(confidence.item())}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import clean_text
from src.inference_logic import hybrid_predict

app = FastAPI(title="Policy Comment Analysis API", description="API for analyzing policy draft comments.")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load models and encoders
# Use absolute paths to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # policy_analysis
PROJECT_ROOT = os.path.dirname(BASE_DIR) # nlp_proj

MODEL_DIR = os.path.join(PROJECT_ROOT, "policy_analysis", "models")
# Improved v2 model
MODEL_DIR_BALANCED = os.path.join(PROJECT_ROOT, "policy_analysis", "models_improved", "sentiment_model_v2")
ENCODER_PATH_BALANCED = os.path.join(PROJECT_ROOT, "policy_analysis", "models_improved", "sentiment_encoder_improved.joblib")

# Fallback for testing if balanced model doesn't exist
MODEL_DIR_TEST = "policy_analysis/models_test/sentiment_model_test"
ENCODER_PATH_TEST = "policy_analysis/models_test/sentiment_encoder_test.joblib"

models = {}
encoders = {}
vectorizers = {}
tokenizers = {}

TASKS = ['sentiment', 'stance', 'topic']

@app.on_event("startup")
def load_artifacts():
    # Load BERT for Sentiment
    try:
        if os.path.exists(MODEL_DIR_BALANCED):
            print(f"Loading BERT model from {MODEL_DIR_BALANCED}")
            models['sentiment'] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_BALANCED)
            tokenizers['sentiment'] = AutoTokenizer.from_pretrained(MODEL_DIR_BALANCED)
            encoders['sentiment'] = joblib.load(ENCODER_PATH_BALANCED)
        elif os.path.exists(MODEL_DIR_TEST):
            print(f"Loading Test BERT model from {MODEL_DIR_TEST}")
            models['sentiment'] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_TEST)
            tokenizers['sentiment'] = AutoTokenizer.from_pretrained(MODEL_DIR_TEST)
            encoders['sentiment'] = joblib.load(ENCODER_PATH_TEST)
        else:
            print("Warning: No BERT model found for sentiment.")
            
        if 'sentiment' in models and torch.cuda.is_available():
            models['sentiment'].to('cuda')
            
    except Exception as e:
        print(f"Error loading BERT model: {e}")

    # Load Sklearn models for Stance and Topic
    for task in ['stance', 'topic']:
        try:
            models[task] = joblib.load(f"{MODEL_DIR}/{task}_model.joblib")
            encoders[task] = joblib.load(f"{MODEL_DIR}/{task}_encoder.joblib")
            vectorizers[task] = joblib.load(f"{MODEL_DIR}/{task}_tfidf.joblib")
            print(f"Loaded artifacts for {task}")
        except FileNotFoundError:
            print(f"Warning: Artifacts for {task} not found.")

from typing import Optional

class CommentRequest(BaseModel):
    text: str
    policy: Optional[str] = None

class PredictionResponse(BaseModel):
    sentiment: str
    stance: str
    topic: str

@app.post("/predict", response_model=PredictionResponse)
def predict(request: CommentRequest):
    cleaned_text = clean_text(request.text)
    response = {}
    
    # Sentiment (BERT)
    if 'sentiment' in models:
        try:
            tokenizer = tokenizers['sentiment']
            model = models['sentiment']
            encoder = encoders['sentiment']
            
            inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            # Use Shared Hybrid Logic
            decision, _, _ = hybrid_predict(probs, request.text)
            response['sentiment'] = decision
        except Exception as e:
            print(f"Error predicting sentiment: {e}")
            response['sentiment'] = "Error"
    else:
        response['sentiment'] = "Model not loaded"

    # Stance and Topic (Sklearn)
    for task in ['stance', 'topic']:
        if task not in models:
            response[task] = "Model not loaded"
            continue
            
        try:
            vectorizer = vectorizers[task]
            model = models[task]
            encoder = encoders[task]
            
            # Transform and predict
            features = vectorizer.transform([cleaned_text])
            prediction_idx = model.predict(features)[0]
            prediction_label = encoder.inverse_transform([prediction_idx])[0]
            
            response[task] = prediction_label
        except Exception as e:
            print(f"Error predicting {task}: {e}")
            response[task] = "Error"
        
    return response

from fastapi.staticfiles import StaticFiles

# ... [imports] ...

# WEB DIR
WEB_DIR = os.path.join(PROJECT_ROOT, "policy_analysis", "web")
print(f"DEBUG: Mounting Static Files from {WEB_DIR}")
if not os.path.exists(WEB_DIR):
    print("CRITICAL ERROR: WEB_DIR DOES NOT EXIST")

# Mount Static Files (Frontend)
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")

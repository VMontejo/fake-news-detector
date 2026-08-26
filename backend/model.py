import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

#Path to the model folder
MODEL_PATH = Path("model/fake_news_model")

#Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model =AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print(f"Model loaded on: {device}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading the model!: {e}")
        return None, None

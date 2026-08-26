import re

def preprocess_text(text):
    """Clean and preprocess text for the model"""

    #Step 1: Lowercase
    text = text.lower()

    #Step 2: Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #Step 3: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

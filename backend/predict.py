import torch
from .preprocess import preprocess_text

def predict_article(model, tokenizer, title, text):
    """Predict if news article is Fake (0) or Real(1)"""

    #Combine title and text
    full_text = title + " " + text

    #Preprocess
    cleaned_text = preprocess_text(full_text)

    #Tokenize
    inputs = tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=False
    )

    #Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    #Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][prediction].item()

    #Map prediction
    label_map = {0: "Fake", 1: "Real"}

    return {
        "label": prediction,
        "prediction": label_map[prediction],
        "confidence": round(confidence *100, 2)
    }

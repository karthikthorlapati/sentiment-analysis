
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import os

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

def predict(text, model, tokenizer, apply_cleaning=False):
    if apply_cleaning:
        processed_text = clean_text(text)
    else:
        processed_text = text
        
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    confidence, predicted_class = torch.max(probs, dim=1)
    labels = ["negative", "positive"]
    sentiment = labels[predicted_class.item()]
    
    print(f"Input: '{text}'")
    print(f"Processed: '{processed_text}'")
    print(f"Sentiment: {sentiment}, Confidence: {confidence.item():.4f}")
    print("-" * 30)

def main():
    model_path = "model_output"
    print(f"Loading model from {model_path}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    test_cases = [
        "i am happy",
        "I am happy!",
        "This is a wonderful movie.",
        "I hate this.",
        "i am strictly neutral"
    ]

    with open("repro_results.txt", "w") as f:
        f.write("\n--- Testing WITHOUT Cleaning (Current API behavior) ---\n")
        for text in test_cases:
            res = predict_str(text, model, tokenizer, apply_cleaning=False)
            f.write(res + "\n")

        f.write("\n--- Testing WITH Cleaning (Preprocess behavior) ---\n")
        for text in test_cases:
            res = predict_str(text, model, tokenizer, apply_cleaning=True)
            f.write(res + "\n")

def predict_str(text, model, tokenizer, apply_cleaning=False):
    if apply_cleaning:
        processed_text = clean_text(text)
    else:
        processed_text = text
        
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    confidence, predicted_class = torch.max(probs, dim=1)
    labels = ["negative", "positive"]
    sentiment = labels[predicted_class.item()]
    
    return f"Input: '{text}' | Processed: '{processed_text}' | Sentiment: {sentiment} | Confidence: {confidence.item():.4f}"


if __name__ == "__main__":
    main()

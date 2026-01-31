
import argparse
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def batch_predict(input_file, output_file, model_path):
    logger.info(f"Loading input file: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return

    if 'text' not in df.columns:
        logger.error("Input file must contain a 'text' column.")
        return

    logger.info(f"Loading model from {model_path}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Please ensure the model is trained and saved in the specified path.")
        return

    logger.info("Running predictions...")
    sentiments = []
    confidences = []
    
    # We can process in batches if the file is large, but for simplicity here we do one by one or small batches.
    # Let's do simple iteration for clarity and error handling
    
    for text in df['text']:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            confidence, predicted_class = torch.max(probs, dim=1)
            labels = ["negative", "positive"]
            sentiments.append(labels[predicted_class.item()])
            confidences.append(confidence.item())
        except Exception as e:
            logger.warning(f"Error predicting text: {text[:50]}... Error: {e}")
            sentiments.append("error")
            confidences.append(0.0)

    df['predicted_sentiment'] = sentiments
    df['confidence'] = confidences
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    logger.info(f"Saving predictions to {output_file}...")
    df.to_csv(output_file, index=False)
    logger.info("Batch prediction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Sentiment Prediction")
    parser.add_argument("--input-file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output-file", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--model-path", type=str, default="model_output", help="Path to trained model")
    
    args = parser.parse_args()
    
    batch_predict(args.input_file, args.output_file, args.model_path)

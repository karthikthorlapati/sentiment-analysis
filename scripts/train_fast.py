
import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch

# Disable problematic optimizations
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    parser = argparse.ArgumentParser(description="Fast Fine-tune DistilBERT for Sentiment Analysis")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test with subsampled data")
    args = parser.parse_args()

    # Configuration - using DistilBERT for faster training on CPU
    model_name = "distilbert-base-uncased"
    output_dir = "model_output"
    results_dir = "results"
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    try:
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")
    except FileNotFoundError:
        logger.error("Data files not found. Please run scripts/preprocess.py first.")
        return

    if args.smoke_test:
        logger.info("Running in SMOKE TEST mode. Subsampling data...")
        train_df = train_df.head(100)
        test_df = test_df.head(20)
        num_epochs = 1
    else:
        # Full training optimized for CPU
        num_epochs = 1
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize
    logger.info("Tokenizing data...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Training Arguments - optimized for CPU
    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,  # DistilBERT is smaller, can handle 8
        per_device_eval_batch_size=16,
        warmup_steps=50 if not args.smoke_test else 5,
        weight_decay=0.01,
        logging_dir=f'{results_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=1,
        log_on_each_node=False,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        learning_rate=2e-5
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate
    logger.info("Evaluating...")
    eval_result = trainer.evaluate()
    
    # Save metrics
    metrics = {
        "accuracy": eval_result["eval_accuracy"],
        "precision": eval_result["eval_precision"],
        "recall": eval_result["eval_recall"],
        "f1_score": eval_result["eval_f1_score"]
    }
    
    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Save Run Summary
    run_summary = {
        "hyperparameters": {
            "model_name": model_name,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "num_epochs": training_args.num_train_epochs
        },
        "final_metrics": metrics
    }
    
    with open(f"{results_dir}/run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    
    # Save Model
    logger.info(f"Saving model to {output_dir}...")
    import shutil
    best_model_path = os.path.join(results_dir, "checkpoint-best")
    if os.path.exists(best_model_path):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(best_model_path, output_dir)
    else:
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        except MemoryError:
            logger.warning("Memory error when saving model, using checkpoint instead...")
            import shutil
            latest_checkpoint = max([d for d in os.listdir(results_dir) if d.startswith("checkpoint-")], 
                                   key=lambda x: int(x.split("-")[1]))
            shutil.copytree(os.path.join(results_dir, latest_checkpoint), output_dir, dirs_exist_ok=True)
    
    logger.info("Training complete.")

if __name__ == "__main__":
    main()

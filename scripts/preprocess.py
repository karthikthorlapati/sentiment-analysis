import os
import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

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

def preprocess_data():
    """
    Load IMDb dataset, clean text, and save to CSV.
    """
    print("Loading IMDb dataset...")
    # Load IMDb dataset from Hugging Face
    dataset = load_dataset("imdb")
    
    # Convert to pandas DataFrames
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    print("Cleaning text...")
    train_df['text'] = train_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].apply(clean_text)

    # Shuffle the data
    print("Shuffling data...")
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    print("Saving processed data...")
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print("Data preprocessing complete.")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

if __name__ == "__main__":
    preprocess_data()

# Minor param optimization
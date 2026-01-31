# Twitter Sentiment Analysis API

This project provides a complete sentiment analysis system using a fine-tuned BERT model on the IMDb dataset (as a proxy for sentiment data). It includes a REST API, a Streamlit UI, and scripts for data processing and training.

## Project Structure

```
├── data/
│   ├── raw/
│   └── processed/
├── model_output/       # Stores fine-tuned model artifacts
├── results/            # Stores evaluation metrics
├── scripts/
│   ├── preprocess.py   # Data cleaning and splitting
│   ├── batch_predict.py # Batch prediction script
│   └── train.py        # Model fine-tuning script
├── src/
│   ├── api.py          # FastAPI application
│   └── ui.py           # Streamlit application
├── tests/
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
└── requirements.txt
```

## Setup

1.  **Environment Variables**:
    Copy `.env.example` to `.env`:
    ```bash
    cp .env.example .env
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preprocessing

Prepare the dataset:
```bash
python scripts/preprocess.py
```
This will download the IMDb dataset and save processed CSV files to `data/processed/`.

### 2. Model Training

Fine-tune the BERT model:
```bash
python scripts/train.py
```
This will train the model and save artifacts to `model_output/`. **This step is required before building the Docker image for the API.**

### 3. Batch Prediction

Run predictions on a CSV file:
```bash
python scripts/batch_predict.py --input-file data/processed/test.csv --output-file results/predictions.csv
```

### 4. Running with Docker

Build and run the services:
```bash
docker-compose up --build
```

Access the services:
- **API**: `http://localhost:8000`
    - Docs: `http://localhost:8000/docs`
    - Health Check: `http://localhost:8000/health`
- **UI**: `http://localhost:8501`

## API Endpoints

-   `GET /health`: Check service status.
-   `POST /predict`: Predict sentiment.
    -   Body: `{"text": "I love this movie!"}`
    -   Response: `{"sentiment": "positive", "confidence": 0.99}`


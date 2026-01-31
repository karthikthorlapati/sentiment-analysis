# Testing Instructions

Follow these steps to test the Sentiment Analysis Project.

## 1. Wait for Setup
Ensure the Docker services are running.
Run:
```bash
docker ps
```
You should see two containers: `sentiment-analysis-api-1` and `sentiment-analysis-ui-1`.

## 2. Test the Web Interface (Easiest)
1.  Open your web browser.
2.  Go to: [http://localhost:8501](http://localhost:8501)
3.  Type a sentence like: "This is a great project!"
4.  Click **Analyze Sentiment**.
5.  You should see "POSITIVE" and a confidence score.

## 3. Test the API (Background Service)
1.  Open a new terminal window.
2.  Check if the API is alive:
    ```bash
    curl http://localhost:8000/health
    ```
    *Output should be:* `{"status":"ok"}`
3.  Get a prediction:
    ```bash
    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"I am not happy with this result.\"}"
    ```
    *Output should be:* `{"sentiment":"negative", "confidence": ...}`

## 4. Test Batch Prediction (Script)
Run the script to predict sentiments for a file of texts:
```bash
python scripts/batch_predict.py --input-file data/processed/test.csv --output-file results/my_test_results.csv
```
Check the results:
```bash
type results\my_test_results.csv
```

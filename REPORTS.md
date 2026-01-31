# Sentiment Analysis Project Details

## Model Choice: DistilBERT
- **Selected Model:** `distilbert-base-uncased`
- **Rationale:** 
  - Retains ~97% of BERT's performance with 40% fewer parameters.
  - 60% faster inference, which is critical for CPU-based API performance.
  - Allows larger batch sizes (e.g., 8 instead of 4) on standard consumer hardware.

## Text Preprocessing implementation
The preprocessing pipeline in `scripts/preprocess.py` includes:
1. **HTML Removal:** Uses regex to strip tags like `<br />` from IMDb reviews.
2. **Noise Reduction:** Removes non-alphanumeric characters to simplify the vocabulary.
3. **Normalization:** Converts text to lowercase to match the `uncased` model requirements.
4. **Trimming:** Strips leading/trailing whitespace for consistent tokenization.

## Hyperparameter Tuning Strategy
- **Current Setup:** Optimized for fast convergence with `lr=2e-5` and `epochs=1`.
- **Planned Strategy:** 
  - Implementation of **Optuna** for Bayesian optimization.
  - Search space: Learning rate ($1e-5$ to $5e-5$), Weight Decay ($0.01$ to $0.1$), and Batch Size (16, 32).
  - Metric: Validation F1-score to handle potential class imbalances.

## Deployment Trade-offs
- **Latency:** CPU inference takes ~150ms. Real-time requirements may necessitate ONNX conversion.
- **Cost:** GPU instances for 24/7 inference can be expensive; batch processing or auto-scaling is preferred.
- **Scalability:** The FastAPI backend is asynchronous, but model execution is compute-bound.

## Scaling to 1 million requests/day
To handle this load (~12 requests/second average, higher peaks):
1. **Infrastructure:** Kubernetes with Horizontal Pod Autoscaler (HPA) based on GPU metrics.
2. **Inference Server:** Use **NVIDIA Triton Inference Server** with dynamic batching.
3. **Caching:** Implement a **Redis** layer for common queries.
4. **Optimization:** Quantize the model to INT8 or use FP16 for 2x-4x speedups.

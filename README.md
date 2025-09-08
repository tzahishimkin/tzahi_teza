# Crocs RTB Relevance System

A real-time bidding (RTB) relevance system for Crocs campaigns. The system analyzes URL snippets against Crocs creative briefs to make bid/no-bid decisions with relevance-based CPM pricing.

## Features

- Sub-100ms latency
- Supports up to 20k QPS
- Sentence transformer-based relevance scoring
- FastAPI REST API endpoint

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and Evaluate Model

```bash
python src/eval.py
```

This will train the relevance model and output evaluation results to `results.json`.

### 3. Start FastAPI Server

```bash
python src/serve.py
```

Or using uvicorn directly:

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with a POST endpoint at `/score`.

## Project Structure

```
.
├── src/
│   ├── model.py          # RelevanceModel class
│   ├── eval.py           # Training and evaluation script
│   └── serve.py          # FastAPI application
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore patterns
├── README.md            # This file
├── results.json         # Model evaluation results (generated)
└── PUBLIC_URL.txt       # Deployment URL (if applicable)
```

## Retrain after encoder change

If you've changed the encoder in `src/model.py`, retrain the model:

```bash
make setup
make retrain
```

Expected outputs:
- `artifacts/` recreated with a fresh model (e.g., model.pkl)
- Console prints test metrics and results summary
- `results.json` regenerated at repo root

## API Usage

```bash
curl -X POST "http://localhost:8000/score" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com", "snippet": "comfortable shoes for outdoor activities"}'
```

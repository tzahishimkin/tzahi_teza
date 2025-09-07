"""
FastAPI service for Crocs RTB relevance scoring.

Run command: uvicorn src.serve:app --host 0.0.0.0 --port 8000
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from model import RelevanceModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model: RelevanceModel = None

# Request/Response models
class ScoreRequest(BaseModel):
    url: str
    snippet: str

class ScoreResponse(BaseModel):
    url: str
    bid: int
    price: float
    score: float

class HealthResponse(BaseModel):
    ok: bool
    model_loaded: bool
    threshold: float
    min_cpm: float
    max_cpm: float

# Create FastAPI app
app = FastAPI(
    title="Crocs RTB Relevance API",
    description="Real-time bidding relevance scoring for Crocs campaigns",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load the trained model on startup."""
    global model
    
    try:
        # Get artifacts directory path
        project_root = Path(__file__).parent.parent
        artifacts_dir = project_root / "artifacts"
        
        if not artifacts_dir.exists():
            logger.error(f"Artifacts directory not found: {artifacts_dir}")
            logger.error("Please run 'python src/eval.py' to train the model first.")
            sys.exit(1)
        
        model_file = artifacts_dir / "model.pkl"
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            logger.error("Please run 'python src/eval.py' to train the model first.")
            sys.exit(1)
        
        # Load the model
        model = RelevanceModel()
        model.load(str(artifacts_dir))
        
        logger.info(f"✅ Model loaded successfully from {artifacts_dir}")
        logger.info(f"   - Threshold: {model.threshold:.3f}")
        logger.info(f"   - CPM range: ${model.min_cpm:.2f} - ${model.max_cpm:.2f}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint returning model status and configuration.
    """
    return HealthResponse(
        ok=True,
        model_loaded=model is not None,
        threshold=model.threshold if model else 0.0,
        min_cpm=model.min_cpm if model else 0.0,
        max_cpm=model.max_cpm if model else 0.0
    )

@app.post("/score", response_model=ScoreResponse)
async def score_snippet(request: ScoreRequest) -> ScoreResponse:
    """
    Score a URL/snippet pair for relevance and return bid decision.
    
    Args:
        request: ScoreRequest containing url and snippet
        
    Returns:
        ScoreResponse with bid decision, price, and relevance score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get prediction from model
        prediction = model.predict(request.snippet)
        
        # Return the response
        return ScoreResponse(
            url=request.url,
            bid=prediction["bid"],
            price=prediction["price"],
            score=prediction["score"]
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend() -> str:
    """
    Serve a simple HTML interface for testing the scoring API.
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crocs RTB Relevance Scorer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #ff6b35;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        input[type="text"], textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            box-sizing: border-box;
        }
        textarea {
            height: 100px;
            resize: vertical;
        }
        button {
            background-color: #ff6b35;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: #e55a2b;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 6px;
            display: none;
        }
        .result.success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .result.error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .bid-yes {
            color: #28a745;
            font-weight: bold;
        }
        .bid-no {
            color: #dc3545;
            font-weight: bold;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .metric {
            text-align: center;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .metric-value {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐊 Crocs RTB Relevance Scorer</h1>
        
        <form id="scoreForm">
            <div class="form-group">
                <label for="url">URL:</label>
                <input type="text" id="url" name="url" 
                       placeholder="https://example.com/article-about-shoes" 
                       value="https://example.com/comfortable-work-shoes">
            </div>
            
            <div class="form-group">
                <label for="snippet">Snippet:</label>
                <textarea id="snippet" name="snippet" 
                          placeholder="Enter the text snippet to evaluate for Crocs campaign relevance...">Nurses recommend lightweight, comfortable Crocs for long hospital shifts</textarea>
            </div>
            
            <button type="submit" id="scoreBtn">Score Relevance</button>
        </form>
        
        <div id="result" class="result">
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        document.getElementById('scoreForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const button = document.getElementById('scoreBtn');
            const result = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            
            // Get form data
            const url = document.getElementById('url').value.trim();
            const snippet = document.getElementById('snippet').value.trim();
            
            if (!url || !snippet) {
                showResult('error', 'Please fill in both URL and snippet fields.');
                return;
            }
            
            // Disable button and show loading
            button.disabled = true;
            button.textContent = 'Scoring...';
            
            try {
                const response = await fetch('/score', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ url, snippet })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    const bidText = data.bid === 1 ? 'BID' : 'NO BID';
                    const bidClass = data.bid === 1 ? 'bid-yes' : 'bid-no';
                    
                    resultContent.innerHTML = `
                        <h3>Prediction Result</h3>
                        <p><strong>Decision:</strong> <span class="${bidClass}">${bidText}</span></p>
                        
                        <div class="metrics">
                            <div class="metric">
                                <div class="metric-value">${(data.score * 100).toFixed(1)}%</div>
                                <div class="metric-label">Relevance Score</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">$${data.price.toFixed(2)}</div>
                                <div class="metric-label">CPM Price</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${data.bid}</div>
                                <div class="metric-label">Bid Flag</div>
                            </div>
                        </div>
                        
                        <p style="margin-top: 15px;"><strong>URL:</strong> ${data.url}</p>
                    `;
                    showResult('success');
                } else {
                    showResult('error', `Error: ${data.detail || 'Unknown error'}`);
                }
            } catch (error) {
                showResult('error', `Network error: ${error.message}`);
            } finally {
                // Re-enable button
                button.disabled = false;
                button.textContent = 'Score Relevance';
            }
        });
        
        function showResult(type, message = '') {
            const result = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            
            result.className = `result ${type}`;
            result.style.display = 'block';
            
            if (message) {
                resultContent.innerHTML = `<p>${message}</p>`;
            }
            
            // Scroll to result
            result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    </script>
</body>
</html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

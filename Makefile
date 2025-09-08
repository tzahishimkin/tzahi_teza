.PHONY: setup train eval all serve retrain

# Setup virtual environment and install dependencies
setup:
	python -m venv venv
	venv/bin/pip install -r requirements.txt

# Train model only
train:
	python -m src.eval --train-only

# Evaluate model only
eval:
	python -m src.eval --eval-only

# Full pipeline (train + eval + results)
all:
	python -m src.eval

# Retrain with clean artifacts (for encoder changes)
retrain:
	python -m src.eval --all --clean

# Start FastAPI server
serve:
	uvicorn src.serve:app --host 0.0.0.0 --port 8000

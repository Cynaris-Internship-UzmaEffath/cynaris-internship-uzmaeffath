# W4D4: FastAPI Model Serving Endpoint

## Objective
Serve a trained Random Forest classifier through FastAPI.

## Model Details
- Dataset: Breast Cancer Wisconsin
- Algorithm: Random Forest Classifier
- Model reused from W4D3 model serialisation
- Input: 30 numeric features
- Output: Predicted class, label, and probabilities

## API Endpoints
- GET /health: Check API health
- POST /predict: Generate a prediction

## Setup
Install dependencies using python -m pip install -r .\W4D4-fastapi-model-serving\requirements.txt.

## Run the API
python -m uvicorn main:app --app-dir .\W4D4-fastapi-model-serving --reload

Open http://127.0.0.1:8000/docs to test the API.

## Run Tests
python -m pytest .\W4D4-fastapi-model-serving\test_main.py -v

## Test Coverage
- Health endpoint
- Valid predictions
- Invalid feature count validation

Three automated tests passed during initial verification.

## Self-Review
- [x] Reused the saved model
- [x] Added health and prediction endpoints
- [x] Added input validation
- [x] Added automated tests
- [ ] Saved final test evidence
- [ ] Reviewed Git changes
- [ ] Pushed branch and opened pull request

## Limitation
This is an educational demonstration, not a medical diagnostic tool.

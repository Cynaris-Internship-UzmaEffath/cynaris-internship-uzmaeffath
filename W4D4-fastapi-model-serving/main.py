"""FastAPI endpoint for serving the saved breast cancer classifier."""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "W4D3-model-serialisation"
    / "outputs"
    / "model_joblib.pkl"
)

# Load the trained model once when the API starts.
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Predicts breast cancer class using a trained Random Forest model.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    """Input features for a single prediction."""

    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 numeric features from the breast cancer dataset.",
    )


@app.get("/health")
def health_check():
    """Check whether the API and model are available."""
    return {"status": "healthy", "model": type(model).__name__}


@app.post("/predict")
def predict(request: PredictionRequest):
    """Return the predicted class and class probabilities."""
    try:
        prediction = int(model.predict([request.features])[0])
        probabilities = model.predict_proba([request.features])[0]

        labels = {0: "malignant", 1: "benign"}

        return {
            "prediction": prediction,
            "predicted_label": labels[prediction],
            "probabilities": {
                str(int(class_id)): round(float(probability), 4)
                for class_id, probability in zip(model.classes_, probabilities)
            },
        }
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="The supplied features could not be processed.",
        ) from exc

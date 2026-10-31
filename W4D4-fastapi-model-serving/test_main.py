"""Tests for the FastAPI model-serving endpoint."""

from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    # Use a real sample with the 30 expected input features.
    sample = load_breast_cancer().data[0].tolist()

    response = client.post("/predict", json={"features": sample})

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["predicted_label"] in ["malignant", "benign"]
    assert set(result["probabilities"]) == {"0", "1"}
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.001


def test_prediction_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [1.0, 2.0]})

    assert response.status_code == 422

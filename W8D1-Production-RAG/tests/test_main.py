import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Production RAG API is running"


@patch("app.main.requests.get")
def test_health_endpoint_healthy(mock_get):
    mock_get.return_value = Mock(
        status_code=200,
        json=lambda: {"models": [{"name": "llama3.2:3b"}]},
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_available"] is True


@patch("app.main.requests.post")
def test_generate_endpoint(mock_post):
    mock_post.return_value = Mock(
        status_code=200,
        json=lambda: {
            "response": "RAG combines retrieval and generation.",
            "done": True,
        },
    )

    response = client.post(
        "/generate",
        json={"prompt": "Explain RAG"},
    )

    assert response.status_code == 200
    assert response.json()["model"] == "llama3.2:3b"
    assert response.json()["response"] == "RAG combines retrieval and generation."
    assert response.json()["done"] is True


def test_generate_rejects_empty_prompt():
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422

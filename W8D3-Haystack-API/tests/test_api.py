import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = ROOT_DIR / "W8D3-Haystack-API" / "src"

sys.path.insert(0, str(APP_DIR))

import app as api


def make_document(content, source="test.pdf"):
    return SimpleNamespace(
        content=content,
        meta={"source": source}
    )


def setup_function():
    api.pipelines.clear()


def test_health_check():
    response = TestClient(api.app).get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["retrievers"] == ["bm25", "dense"]


def test_bm25_search(monkeypatch):
    documents = [make_document("Machine learning basics")]

    class FakePipeline:
        def run(self, data):
            assert "retriever" in data
            return {"retriever": {"documents": documents}}

    monkeypatch.setitem(api.pipelines, "bm25", FakePipeline())

    response = TestClient(api.app).get(
        "/search",
        params={"q": "machine learning", "method": "bm25"}
    )

    assert response.status_code == 200
    assert response.json()["method"] == "bm25"
    assert response.json()["results"][0]["source"] == "test.pdf"


def test_dense_search(monkeypatch):
    documents = [make_document("Semantic search results")]

    class FakePipeline:
        def run(self, data):
            assert "text_embedder" in data
            return {"retriever": {"documents": documents}}

    monkeypatch.setitem(api.pipelines, "dense", FakePipeline())

    response = TestClient(api.app).get(
        "/search",
        params={"q": "semantic search", "method": "dense"}
    )

    assert response.status_code == 200
    assert response.json()["method"] == "dense"
    assert response.json()["results"][0]["content"] == "Semantic search results"


def test_invalid_method():
    response = TestClient(api.app).get(
        "/search",
        params={"q": "machine learning", "method": "invalid"}
    )

    assert response.status_code == 400


def test_short_query():
    response = TestClient(api.app).get(
        "/search",
        params={"q": "a"}
    )

    assert response.status_code == 422


def test_answer_endpoint(monkeypatch):
    documents = [
        make_document("First document"),
        make_document("Second document", "cloud.pdf")
    ]

    class FakePipeline:
        def run(self, data):
            return {"retriever": {"documents": documents}}

    monkeypatch.setitem(api.pipelines, "bm25", FakePipeline())

    response = TestClient(api.app).get(
        "/answer",
        params={"q": "cloud computing"}
    )

    assert response.status_code == 200
    assert "First document" in response.json()["context"]
    assert "Second document" in response.json()["context"]
    assert response.json()["sources"] == ["test.pdf", "cloud.pdf"]


def test_answer_with_no_documents(monkeypatch):
    class FakePipeline:
        def run(self, data):
            return {"retriever": {"documents": []}}

    monkeypatch.setitem(api.pipelines, "bm25", FakePipeline())

    response = TestClient(api.app).get(
        "/answer",
        params={"q": "unknown topic"}
    )

    assert response.status_code == 200
    assert response.json()["context"] == ""
    assert response.json()["sources"] == []

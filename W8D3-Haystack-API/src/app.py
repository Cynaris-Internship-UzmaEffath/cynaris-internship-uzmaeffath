import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

ROOT_DIR = Path(__file__).resolve().parents[2]
W7D2_DIR = ROOT_DIR / "W7D2-Haystack-Retrieval"

sys.path.insert(0, str(W7D2_DIR))

from bm25_pipeline import build_bm25_pipeline
from dense_pipeline import build_dense_pipeline


app = FastAPI(
    title="Haystack Retrieval API",
    description="PDF question retrieval using BM25 and Dense Retrieval",
    version="1.0.0"
)

pipelines = {}


@app.on_event("startup")
def initialize_pipelines():
    pipelines["bm25"] = build_bm25_pipeline()
    pipelines["dense"] = build_dense_pipeline()


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Haystack Retrieval API",
        "retrievers": ["bm25", "dense"]
    }


def retrieve_documents(question: str, method: str):
    if method not in pipelines:
        raise HTTPException(
            status_code=400,
            detail="Method must be bm25 or dense"
        )

    if method == "bm25":
        result = pipelines["bm25"].run({
            "retriever": {"query": question}
        })
        documents = result["retriever"]["documents"]

    else:
        result = pipelines["dense"].run({
            "text_embedder": {"text": question}
        })
        documents = result["retriever"]["documents"]

    return documents


@app.get("/search")
def search(
    q: str = Query(..., min_length=2),
    method: str = Query("bm25"),
):
    documents = retrieve_documents(q, method)

    return {
        "question": q,
        "method": method,
        "results": [
            {
                "rank": index,
                "source": doc.meta.get("source", "Unknown"),
                "content": doc.content[:500]
            }
            for index, doc in enumerate(documents, start=1)
        ]
    }


@app.get("/answer")
def answer(
    q: str = Query(..., min_length=2),
    method: str = Query("bm25"),
):
    documents = retrieve_documents(q, method)

    if not documents:
        return {
            "question": q,
            "method": method,
            "context": "",
            "sources": []
        }

    return {
        "question": q,
        "method": method,
        "context": "\n\n".join(
            doc.content for doc in documents
        ),
        "sources": [
            doc.meta.get("source", "Unknown")
            for doc in documents
        ]
    }

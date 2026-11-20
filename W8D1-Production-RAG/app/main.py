import os

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Production RAG API",
    description="Local Ollama-powered API",
    version="1.0.0"
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = "llama3.2:3b"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


@app.get("/")
def root():
    return {"message": "Production RAG API is running"}


@app.get("/health")
def health():
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=10
        )
        response.raise_for_status()

        models = response.json().get("models", [])
        available = any(
            model.get("name") == MODEL_NAME
            for model in models
        )

        return {
            "status": "healthy" if available else "degraded",
            "ollama_connected": True,
            "model": MODEL_NAME,
            "model_available": available
        }

    except requests.RequestException:
        return {
            "status": "degraded",
            "ollama_connected": False,
            "model": MODEL_NAME,
            "model_available": False
        }


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": request.prompt,
                "stream": False
            },
            timeout=180
        )
        response.raise_for_status()

        result = response.json()

        return {
            "model": MODEL_NAME,
            "response": result.get("response", ""),
            "done": result.get("done", False)
        }

    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Ollama request timed out"
        )

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama service error: {exc}"
        )

"""Evaluate the local research report using Ragas answer relevancy."""

import json
from pathlib import Path

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_FILE = OUTPUT_DIR / "research_report.md"
RESULT_FILE = OUTPUT_DIR / "evaluation_results.json"

TOPIC = "Applications of artificial intelligence in cybersecurity"


def main():
    if not REPORT_FILE.exists():
        raise FileNotFoundError(
            f"Research report not found: {REPORT_FILE}"
        )

    report = REPORT_FILE.read_text(encoding="utf-8").strip()

    if not report:
        raise ValueError("Research report is empty.")

    dataset = Dataset.from_dict({
        "user_input": [TOPIC],
        "response": [report],
    })

    local_llm = ChatOllama(
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    local_embeddings = OllamaEmbeddings(
        model="all-minilm:latest",
        base_url="http://localhost:11434",
    )

    print("Starting local Ragas answer relevancy evaluation...")

    results = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy],
        llm=local_llm,
        embeddings=local_embeddings,
        raise_exceptions=True,
        show_progress=True,
    )

    score = results["answer_relevancy"]

    evaluation = {
        "topic": TOPIC,
        "metric": "answer_relevancy",
        "score": float(score[0]),
        "llm_model": "llama3.2:3b",
        "embedding_model": "all-minilm:latest",
        "evaluation_provider": "Local Ollama",
        "limitations": [
            "The assistant generates reports without retrieving source documents.",
            "Faithfulness and context-based retrieval metrics are not evaluated.",
            "The report is generated content and is not independently source-verified."
        ],
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULT_FILE.write_text(
        json.dumps(evaluation, indent=4),
        encoding="utf-8",
    )

    print("\n===== RAGAS EVALUATION =====")
    print(f"Answer relevancy: {score[0]:.4f}")
    print(f"Results saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()






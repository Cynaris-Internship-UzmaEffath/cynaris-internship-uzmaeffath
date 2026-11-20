
import json
import chromadb
import ollama

from pathlib import Path
from qa_bot import (
    DB_PATH,
    COLLECTION_NAME,
    retrieve_context,
    generate_answer
)

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

QUESTIONS = [
    "What is machine learning and how does it help AI?",
    "How does ChromaDB work with Ollama in a RAG system?",
    "How does intrusion detection help cybersecurity?"
]

def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    results = {}

    for question in QUESTIONS:
        print("\n" + "=" * 60)
        print("Question:", question)

        context = retrieve_context(question, collection)

        results[question] = {}

        for model in MODELS:
            print(f"\nTesting {model}...")

            answer = generate_answer(
                question,
                context,
                model
            )

            results[question][model] = answer

            print("Answer:", answer)

    output_path = Path(__file__).parent / "comparison_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\nComparison completed!")
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

from bm25_pipeline import build_bm25_pipeline
from evaluation_questions import evaluation_questions


def evaluate_bm25():
    pipeline = build_bm25_pipeline()

    total_hits = 0
    top1_correct = 0
    total_questions = len(evaluation_questions)

    evaluation_results = []

    for index, item in enumerate(evaluation_questions, start=1):
        question = item["question"]
        expected_source = item["expected_source"]

        result = pipeline.run({
            "retriever": {
                "query": question
            }
        })

        documents = result["retriever"]["documents"]
        retrieved_sources = [
            document.meta.get("source", "Unknown")
            for document in documents
        ]

        hits = sum(
            1 for source in retrieved_sources
            if source == expected_source
        )

        total_hits += hits

        if retrieved_sources and retrieved_sources[0] == expected_source:
            top1_correct += 1

        evaluation_results.append({
            "question": question,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "correct_source_found": expected_source in retrieved_sources
        })

        print(f"\nQuestion {index}: {question}")
        print("Expected:", expected_source)
        print("Retrieved:", retrieved_sources)

    precision_at_3 = total_hits / (total_questions * 3)
    hit_rate_at_3 = sum(
        1 for item in evaluation_results
        if item["correct_source_found"]
    ) / total_questions
    top1_accuracy = top1_correct / total_questions

    summary = {
        "retrieval_method": "BM25",
        "total_questions": total_questions,
        "precision_at_3": round(precision_at_3, 4),
        "hit_rate_at_3": round(hit_rate_at_3, 4),
        "top1_accuracy": round(top1_accuracy, 4),
        "results": evaluation_results
    }

    output_path = Path("W7D2-Haystack-Retrieval/results/bm25_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nBM25 Evaluation Summary")
    print("-----------------------")
    print("Precision@3:", summary["precision_at_3"])
    print("Hit Rate@3:", summary["hit_rate_at_3"])
    print("Top-1 Accuracy:", summary["top1_accuracy"])
    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    evaluate_bm25()

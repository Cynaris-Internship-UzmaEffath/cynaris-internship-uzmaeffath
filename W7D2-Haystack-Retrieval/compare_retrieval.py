import json
from pathlib import Path


RESULTS_DIR = Path("W7D2-Haystack-Retrieval/results")


def load_results(filename):
    with (RESULTS_DIR / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


def compare_retrieval():
    bm25 = load_results("bm25_results.json")
    dense = load_results("dense_results.json")

    comparison = {
        "dataset": "Five educational PDF documents",
        "total_questions": bm25["total_questions"],
        "bm25": {
            "precision_at_3": bm25["precision_at_3"],
            "hit_rate_at_3": bm25["hit_rate_at_3"],
            "top1_accuracy": bm25["top1_accuracy"]
        },
        "dense": {
            "precision_at_3": dense["precision_at_3"],
            "hit_rate_at_3": dense["hit_rate_at_3"],
            "top1_accuracy": dense["top1_accuracy"]
        }
    }

    output_path = RESULTS_DIR / "comparison_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4)

    print("\nBM25 vs Dense Retrieval")
    print("-----------------------")
    print(f"{'Metric':<20}{'BM25':<15}{'Dense':<15}")

    metrics = [
        ("Precision@3", "precision_at_3"),
        ("Hit Rate@3", "hit_rate_at_3"),
        ("Top-1 Accuracy", "top1_accuracy")
    ]

    for label, key in metrics:
        print(
            f"{label:<20}"
            f"{bm25[key] if key in bm25 else bm25.get(key, 'N/A'):<15}"
            f"{dense[key] if key in dense else dense.get(key, 'N/A'):<15}"
        )

    print("\nComparison saved to:", output_path)


if __name__ == "__main__":
    compare_retrieval()

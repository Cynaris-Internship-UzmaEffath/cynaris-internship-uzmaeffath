import json
from pathlib import Path


def load_results(filename):
    with (Path("results") / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


bm25 = load_results("bm25_results.json")
dense = load_results("dense_results.json")

comparison = {
    "dataset": "Five educational PDF documents",
    "total_questions": bm25["total_questions"],
    "metrics": {
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
    },
    "question_wise_results": []
}

for index, (b, d) in enumerate(
    zip(bm25["results"], dense["results"]), start=1
):
    comparison["question_wise_results"].append({
        "question_number": index,
        "question": b["question"],
        "expected_source": b["expected_source"],
        "bm25_rank": b["relevant_document_rank"],
        "dense_rank": d["relevant_document_rank"]
    })

output_path = Path("results/comparison_results.json")
output_path.write_text(
    json.dumps(comparison, indent=4),
    encoding="utf-8"
)

print("Comparison results saved successfully.")
print(f"File: {output_path}")
print(f"Questions compared: {len(comparison['question_wise_results'])}")

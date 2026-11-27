import json
from pathlib import Path

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

from load_documents import load_pdf_documents
from evaluation_questions import EVALUATION_QUESTIONS


def main():
    documents = load_pdf_documents()

    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    pipeline = Pipeline()
    pipeline.add_component(
        "retriever",
        InMemoryBM25Retriever(
            document_store=document_store,
            top_k=3
        )
    )

    results = []
    total_hits = 0
    top1_correct = 0

    print("\n===== BM25 EVALUATION =====")

    for index, item in enumerate(EVALUATION_QUESTIONS, start=1):
        question = item["question"]
        expected = item["expected_source"]

        response = pipeline.run({
            "retriever": {"query": question}
        })

        retrieved = response["retriever"]["documents"]
        retrieved_sources = [doc.meta["source"] for doc in retrieved]

        hit = expected in retrieved_sources
        total_hits += int(hit)

        rank = retrieved_sources.index(expected) + 1 if hit else None
        top1 = bool(retrieved_sources and retrieved_sources[0] == expected)
        top1_correct += int(top1)

        print(f"\nQuestion {index}: {question}")
        print(f"Expected: {expected}")
        print(f"Retrieved: {retrieved_sources}")
        print(f"Relevant document found: {hit}")
        print(f"Relevant document rank: {rank}")

        results.append({
            "question": question,
            "expected_source": expected,
            "retrieved_sources": retrieved_sources,
            "relevant_document_found": hit,
            "relevant_document_rank": rank,
            "top1_correct": top1
        })

    question_count = len(EVALUATION_QUESTIONS)
    precision_at_3 = total_hits / (question_count * 3)
    hit_rate_at_3 = total_hits / question_count
    top1_accuracy = top1_correct / question_count

    summary = {
        "method": "BM25",
        "total_questions": question_count,
        "relevant_documents_found": total_hits,
        "precision_at_3": round(precision_at_3, 4),
        "hit_rate_at_3": round(hit_rate_at_3, 4),
        "top1_accuracy": round(top1_accuracy, 4),
        "results": results
    }

    output_path = Path("results/bm25_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("\n===== BM25 SUMMARY =====")
    print(f"Questions evaluated: {question_count}")
    print(f"Relevant documents found in top 3: {total_hits}")
    print(f"Precision@3: {precision_at_3:.2%}")
    print(f"Hit Rate@3: {hit_rate_at_3:.2%}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2%}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

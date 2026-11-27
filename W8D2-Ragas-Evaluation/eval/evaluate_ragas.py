import asyncio
import json
from pathlib import Path
from statistics import mean

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "results" / "baseline_answers.json"
OUTPUT_FILE = BASE_DIR / "results" / "baseline_ragas_scores.json"

client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

llm = llm_factory(
    "qwen2.5:3b",
    provider="openai",
    client=client
)

embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    use_api=False,
    device="cpu"
)

faithfulness = Faithfulness(llm=llm)
answer_relevancy = AnswerRelevancy(
    llm=llm,
    embeddings=embeddings
)
context_precision = ContextPrecisionWithReference(llm=llm)
context_recall = ContextRecall(llm=llm)


async def main():
    records = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    results = []

    metric_scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": []
    }

    for record in records:
        question = record["question"]
        answer = record["answer"]
        reference = record["reference"]
        contexts = record["retrieved_contexts"]

        print(f"\nEvaluating {record['id']}: {question}")

        row = {
            "id": record["id"],
            "question": question,
            "scores": {}
        }

        evaluations = [
            (
                "faithfulness",
                faithfulness.ascore(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts
                )
            ),
            (
                "answer_relevancy",
                answer_relevancy.ascore(
                    user_input=question,
                    response=answer
                )
            ),
            (
                "context_precision",
                context_precision.ascore(
                    user_input=question,
                    reference=reference,
                    retrieved_contexts=contexts
                )
            ),
            (
                "context_recall",
                context_recall.ascore(
                    user_input=question,
                    retrieved_contexts=contexts,
                    reference=reference
                )
            )
        ]

        for metric_name, evaluation in evaluations:
            try:
                result = await evaluation
                score = result.value

                row["scores"][metric_name] = score

                if isinstance(score, (int, float)):
                    metric_scores[metric_name].append(score)

                print(f"  {metric_name}: {score}")

            except Exception as error:
                row["scores"][metric_name] = None
                row.setdefault("errors", {})[metric_name] = str(error)
                print(f"  {metric_name} failed: {error}")

        results.append(row)

        OUTPUT_FILE.write_text(
            json.dumps(
                {
                    "evaluation_model": "qwen2.5:3b",
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "total_questions": len(records),
                    "results": results
                },
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    averages = {
        name: round(mean(scores), 4) if scores else None
        for name, scores in metric_scores.items()
    }

    output = {
        "evaluation_model": "qwen2.5:3b",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "total_questions": len(records),
        "metric_averages": averages,
        "results": results
    }

    OUTPUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n========== BASELINE RAGAS RESULTS ==========")

    for name, score in averages.items():
        print(f"{name}: {score}")

    print("\nSaved results to:", OUTPUT_FILE)


asyncio.run(main())

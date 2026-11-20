import json
import time
import urllib.request
from pathlib import Path

API_URL = "http://localhost:11434/api/chat"

MODELS = ["llama3.2:3b", "qwen2.5:3b"]

QUESTIONS = [
    "Explain supervised learning in simple English with one example.",
    "What is quantisation in an LLM, and what is one trade-off?",
    "Give three practical ways to protect a REST API."
]

SYSTEM_PROMPT = (
    "You are a clear and accurate computer science tutor. "
    "Answer in simple English, use practical examples, "
    "and keep each answer under 120 words."
)


def ask_model(model, question):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "stream": False,
        "options": {"temperature": 0.3}
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.perf_counter()

    with urllib.request.urlopen(request, timeout=240) as response:
        data = json.loads(response.read().decode("utf-8"))

    return {
        "model": model,
        "question": question,
        "answer": data["message"]["content"],
        "response_time_seconds": round(time.perf_counter() - start, 2)
    }


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    results = []

    for question in QUESTIONS:
        print(f"\n{'=' * 70}\nQUESTION: {question}")

        for model in MODELS:
            print(f"\n--- {model} ---")

            try:
                result = ask_model(model, question)
                results.append(result)
                print(result["answer"])
                print(f"Time: {result['response_time_seconds']} seconds")
            except Exception as error:
                print(f"Failed for {model}: {error}")

    output_file = output_dir / "model_comparison.json"
    output_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nSaved comparison results to {output_file.resolve()}")
    print(f"Successful responses: {len(results)}/6")


if __name__ == "__main__":
    main()

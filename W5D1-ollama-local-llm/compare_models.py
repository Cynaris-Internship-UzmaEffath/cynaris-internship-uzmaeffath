import json
import urllib.request
from pathlib import Path

API_URL = "http://localhost:11434/api/chat"
MODELS = ["llama3.2:3b", "qwen2.5:3b"]

QUESTIONS = [
    "What is the difference between a model parameter and a prompt parameter?",
    "Why use a local LLM instead of a cloud API?",
    "What is quantisation, and how can it affect model quality?"
]

SYSTEM_PROMPT = (
    "You are an AI/ML tutor. Answer accurately and clearly "
    "using simple language. Keep each answer under 120 words."
)


def ask_model(model, question):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "stream": False
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["message"]["content"]


def main():
    report = [
        "# W5D1: Local LLM Comparison",
        "",
        "Both models received the same system prompt and questions.",
        ""
    ]

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\nQuestion {index}: {question}")
        report.extend([f"## Question {index}", "", question, ""])

        for model in MODELS:
            print(f"\nRunning {model}...")
            answer = ask_model(model, question)
            print(answer)

            report.extend([
                f"### {model}",
                "",
                answer,
                ""
            ])

    output_path = (
        Path(__file__).parent / "outputs" / "model_comparison.md"
    )
    output_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()

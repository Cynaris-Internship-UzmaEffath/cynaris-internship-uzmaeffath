import json
import urllib.request
from pathlib import Path

API_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are a helpful AI/ML learning assistant.
Answer clearly and accurately using simple language.
If you are unsure, say so instead of inventing facts.
"""

PROMPTS = [
    "Explain local LLM inference in simple terms.",
    "What is the difference between a model parameter and a prompt parameter?",
    "Why use a local LLM instead of a cloud API?",
    "What is quantisation in machine learning?",
    "Explain overfitting with a simple example."
]


def ask_ollama(prompt):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
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
    results = [
        "# W5D1: Local LLM Inference",
        "",
        f"Model: `{MODEL}`",
        "",
        "## Custom System Prompt",
        SYSTEM_PROMPT.strip(),
        ""
    ]

    for index, prompt in enumerate(PROMPTS, start=1):
        print(f"\nRunning prompt {index}/{len(PROMPTS)}...")
        answer = ask_ollama(prompt)

        print(answer)
        print("-" * 50)

        results.extend([
            f"## Prompt {index}",
            "",
            f"**Question:** {prompt}",
            "",
            "**Response:**",
            "",
            answer,
            ""
        ])

    output_file = Path(__file__).parent / "outputs" / "first_run_results.md"
    output_file.write_text("\n".join(results), encoding="utf-8")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

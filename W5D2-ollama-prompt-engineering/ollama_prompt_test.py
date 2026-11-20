import json
import time
import urllib.request
import urllib.error
from pathlib import Path

API_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are a helpful AI/ML learning assistant for a computer science student.
Follow these rules:
1. Explain technical concepts in simple English.
2. Use a short example whenever useful.
3. Keep answers under 120 words.
4. If a question is unclear, state your assumption.
5. Do not invent facts or claim certainty when unsure.
"""

PROMPTS = [
    "What is prompt engineering?",
    "Explain the difference between supervised and unsupervised learning.",
    "What is quantisation in a large language model?",
    "Why can a local LLM be useful for cybersecurity projects?",
    "Explain the difference between a system prompt and a user prompt."
]


def ask_ollama(prompt):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
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

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    elapsed = time.perf_counter() - start

    return {
        "model": MODEL,
        "prompt": prompt,
        "system_prompt": SYSTEM_PROMPT.strip(),
        "response": result["message"]["content"],
        "response_time_seconds": round(elapsed, 2)
    }


def main():
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    results = []

    print(f"Model: {MODEL}")
    print("Testing 5 prompts with the custom system prompt.\n")

    for number, prompt in enumerate(PROMPTS, start=1):
        print(f"\n{'=' * 60}")
        print(f"TEST {number}: {prompt}")

        try:
            result = ask_ollama(prompt)
            results.append(result)
            print(f"\nResponse:\n{result['response']}")
            print(f"\nResponse time: {result['response_time_seconds']} seconds")

        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
            print(f"Request failed: {error}")
            print("Check that Ollama is running and the model is available.")

    output_file = output_dir / "prompt_test_results.json"
    output_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nCompleted {len(results)}/{len(PROMPTS)} tests.")
    print(f"Results saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()

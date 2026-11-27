
import json
from pathlib import Path
import ollama

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """
You are a helpful AI and Cybersecurity assistant.
Provide clear, accurate and simple explanations.
Keep answers relevant to the user's question.
"""

PROMPTS = [
    "What is Artificial Intelligence?",
    "What is machine learning?",
    "Explain the role of ChromaDB in a RAG system.",
    "How does intrusion detection improve cybersecurity?",
    "What are the benefits of running an LLM locally?"
]


def main():
    results = []

    print("Ollama Local Inference Test")
    print("Model:", MODEL)

    for index, prompt in enumerate(PROMPTS, start=1):
        print(f"\nTest {index}: {prompt}")

        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response["message"]["content"]

        print("Answer:", answer)

        results.append({
            "prompt": prompt,
            "response": answer
        })

    output_path = Path(__file__).parent / "ollama_responses.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\nAll five prompts completed successfully.")
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()


import json
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

# Models to compare
MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

SYSTEM_PROMPT = """
You are an AI and Machine Learning tutor.
Explain concepts accurately using simple language.
Give practical examples.
Keep answers clear and relevant.
"""

# Identical questions for both models
QUESTIONS = [
    "What is Retrieval-Augmented Generation (RAG)?",
    "How does image classification work in Machine Learning?",
    "How can AI help detect crop diseases?"
]


def ask_model(model, question):

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": False
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    start = time.time()

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    elapsed = time.time() - start

    return result["message"]["content"], elapsed


if __name__ == "__main__":

    results = {}

    for model in MODELS:

        print("\n" + "#" * 65)
        print("MODEL:", model)

        results[model] = []

        for index, question in enumerate(QUESTIONS, start=1):

            print("\n" + "=" * 60)
            print("QUESTION", index)
            print(question)

            try:
                answer, duration = ask_model(model, question)

                print("\nANSWER:")
                print(answer)

                print(f"\nResponse Time: {duration:.2f} seconds")

                results[model].append({
                    "question": question,
                    "answer": answer,
                    "response_time": round(duration, 2)
                })

            except Exception as error:
                print("Error:", error)

    # Save comparison results
    with open("comparison_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\nComparison completed!")
    print("Results saved to comparison_results.json")

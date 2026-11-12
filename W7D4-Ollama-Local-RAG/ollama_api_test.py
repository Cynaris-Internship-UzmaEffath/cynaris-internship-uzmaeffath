
import json
import time
import urllib.request

# Ollama local API
OLLAMA_URL = "http://localhost:11434/api/chat"

# Model to test
MODEL = "llama3.2:3b"

# Custom system prompt
SYSTEM_PROMPT = """
You are a helpful AI and Machine Learning tutor.
Explain technical concepts in simple language.
Give practical examples wherever possible.
Keep answers clear and beginner-friendly.
"""

# Five test prompts
prompts = [
    "What is Artificial Intelligence?",
    "Explain Machine Learning with an example.",
    "What is Deep Learning?",
    "What is Natural Language Processing?",
    "What is Retrieval-Augmented Generation (RAG)?"
]


def ask_ollama(prompt):

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
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

    start_time = time.time()

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    elapsed_time = time.time() - start_time

    return result["message"]["content"], elapsed_time


# Execute all five prompts
if __name__ == "__main__":

    print("OLLAMA LOCAL API TEST")
    print("Model:", MODEL)

    for index, prompt in enumerate(prompts, start=1):

        print("\n" + "=" * 60)
        print(f"TEST {index}")
        print("Question:", prompt)

        try:
            answer, response_time = ask_ollama(prompt)

            print("\nAnswer:")
            print(answer)

            print(f"\nResponse Time: {response_time:.2f} seconds")

        except Exception as error:
            print("Error:", error)

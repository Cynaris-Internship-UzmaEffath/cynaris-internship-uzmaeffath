import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
SIMILARITY_THRESHOLD = 0.92
CACHE_FILE = Path("outputs/semantic_cache.json")

# 20 sample question-answer pairs
qa_pairs = [
    ("What is artificial intelligence?",
     "Artificial intelligence enables machines to perform tasks that normally require human intelligence."),

    ("What is machine learning?",
     "Machine learning allows systems to learn patterns from data."),

    ("What is deep learning?",
     "Deep learning uses neural networks with multiple layers."),

    ("What is cybersecurity?",
     "Cybersecurity protects systems, networks, and data from digital threats."),

    ("What is an intrusion detection system?",
     "An intrusion detection system monitors activities to identify suspicious or malicious behaviour."),

    ("What is network traffic analysis?",
     "Network traffic analysis examines data moving across a network to identify patterns and threats."),

    ("What is a denial of service attack?",
     "A denial of service attack attempts to make a service unavailable by overwhelming it with requests."),

    ("What is phishing?",
     "Phishing is a social engineering attack that tricks users into revealing sensitive information."),

    ("What is a firewall?",
     "A firewall monitors and controls incoming and outgoing network traffic based on security rules."),

    ("What is encryption?",
     "Encryption converts readable information into an encoded format to protect confidentiality."),

    ("What is a neural network?",
     "A neural network is a computing model inspired by interconnected neurons in the human brain."),

    ("What is natural language processing?",
     "Natural language processing enables computers to understand and process human language."),

    ("What is a large language model?",
     "A large language model is trained on large text datasets to understand and generate language."),

    ("What is retrieval augmented generation?",
     "Retrieval augmented generation retrieves relevant external information and uses it as context for language generation."),

    ("What is a vector database?",
     "A vector database stores embeddings and supports similarity searches."),

    ("What are embeddings?",
     "Embeddings are numerical representations of data that capture semantic meaning."),

    ("What is semantic search?",
     "Semantic search retrieves information based on meaning rather than exact keyword matching."),

    ("What is prompt engineering?",
     "Prompt engineering involves designing instructions to guide a language model's responses."),

    ("What is model pruning?",
     "Model pruning removes less important model parameters to reduce model size and computation."),

    ("What is model quantization?",
     "Model quantization reduces numerical precision to make models smaller and more efficient.")
]

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

questions = [item[0] for item in qa_pairs]
answers = [item[1] for item in qa_pairs]

embeddings = model.encode(
    questions,
    normalize_embeddings=True
)

def search_cache(query):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Cosine similarity because embeddings are normalized
    similarities = np.dot(embeddings, query_embedding)

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    if best_score >= SIMILARITY_THRESHOLD:
        return {
            "cache_hit": True,
            "matched_question": questions[best_index],
            "answer": answers[best_index],
            "similarity": round(best_score, 4)
        }

    return {
        "cache_hit": False,
        "matched_question": None,
        "answer": None,
        "similarity": round(best_score, 4)
    }

def save_cache():
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "question": question,
            "answer": answer,
            "embedding": embedding.tolist()
        }
        for question, answer, embedding in zip(
            questions, answers, embeddings
        )
    ]

    CACHE_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )

if __name__ == "__main__":
    save_cache()

    print(f"\nStored {len(qa_pairs)} Q&A pairs.")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    test_queries = [
    "What is artificial intelligence?",
    "What is a firewall?",
    "What is quantum computing?"
    ]
    for query in test_queries:
        result = search_cache(query)

        print(f"\nQuery: {query}")
        print(f"Cache hit: {result['cache_hit']}")
        print(f"Similarity: {result['similarity']}")

        if result["cache_hit"]:
            print(f"Answer: {result['answer']}")
        else:
            print("No sufficiently similar cached answer.")
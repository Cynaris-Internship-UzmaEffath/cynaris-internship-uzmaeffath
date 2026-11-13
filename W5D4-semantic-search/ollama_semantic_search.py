
import json
import urllib.request

import chromadb


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "all-minilm:latest"


def get_embeddings(texts):
    """Generate local embeddings using Ollama."""
    payload = json.dumps({
        "model": EMBEDDING_MODEL,
        "input": texts
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["embeddings"]


client = chromadb.Client()

collection = client.create_collection(
    name="w5d4_ollama_search",
    metadata={"hnsw:space": "cosine"}
)

documents = [
    "ChromaDB stores vector embeddings and supports semantic search.",
    "Ollama runs language models locally on a computer.",
    "Intrusion detection identifies suspicious network activity.",
    "Encryption protects sensitive information from unauthorized access.",
    "Python is widely used for machine learning and data science.",
    "FastAPI provides endpoints for machine learning applications.",
    "Vector databases retrieve documents based on embedding similarity.",
    "All-MiniLM converts text into compact numerical embeddings."
]

ids = [f"doc_{i:02d}" for i in range(1, len(documents) + 1)]

metadatas = [
    {"category": category}
    for category in [
        "database",
        "llm",
        "security",
        "security",
        "programming",
        "development",
        "database",
        "embeddings"
    ]
]

document_embeddings = get_embeddings(documents)

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=document_embeddings
)


def search(query, n_results=3):
    query_embedding = get_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\nQuery: {query}")

    for i, document in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]

        print(f"\nRank: {i + 1}")
        print(f"Document: {document}")
        print(f"Category: {results['metadatas'][0][i]['category']}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Derived cosine similarity: {1 - distance:.4f}")


if __name__ == "__main__":
    print("Embedding model:", EMBEDDING_MODEL)
    print("Embedding dimensions:", len(document_embeddings[0]))
    print("Documents stored:", collection.count())

    search("Finding information about suspicious network attacks")
    search("How do vector databases find similar text?")

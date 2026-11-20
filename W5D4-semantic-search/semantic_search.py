
import chromadb

# Create an in-memory ChromaDB client
client = chromadb.Client()

# Use cosine distance to compare embeddings
collection = client.create_collection(
    name="w5d4_semantic_search",
    metadata={"hnsw:space": "cosine"}
)

# Twenty documents covering different technical topics
documents = [
    "Python is a programming language used in data science.",
    "Java is commonly used to build enterprise applications.",
    "SQL is used to query and manage relational databases.",
    "ChromaDB stores embeddings and supports similarity search.",
    "Vector databases retrieve information using numerical embeddings.",
    "Ollama runs large language models locally.",
    "All-MiniLM converts text into compact semantic embeddings.",
    "Machine learning models learn patterns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Cybersecurity protects systems, networks, and information.",
    "Intrusion detection identifies suspicious network activity.",
    "Encryption protects information by transforming readable data.",
    "FastAPI can expose machine learning models through REST APIs.",
    "Pandas provides tools for manipulating tabular data.",
    "NumPy supports numerical computing in Python.",
    "Git tracks changes to files during software development.",
    "Docker packages applications and their dependencies.",
    "Retrieval augmented generation combines retrieval with LLM answers.",
    "Cosine similarity compares the directions of two vectors.",
    "Metadata filtering restricts search results using document attributes."
]

categories = [
    "programming", "programming", "database", "database",
    "database", "llm", "embeddings", "machine_learning",
    "machine_learning", "security", "security", "security",
    "development", "data_science", "data_science", "development",
    "development", "llm", "embeddings", "database"
]

# Insert documents and metadata. ChromaDB generates embeddings automatically.
collection.add(
    ids=[f"doc_{i:02d}" for i in range(1, 21)],
    documents=documents,
    metadatas=[
        {"category": category, "document_number": i}
        for i, category in enumerate(categories, start=1)
    ]
)


def search(query, category=None, n_results=3):
    """Search documents and optionally filter by category."""
    where = {"category": category} if category else None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\nQuery: {query}")
    if category:
        print(f"Category filter: {category}")

    for i, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        similarity = 1 - distance

        print(f"\n{ i + 1 }. ID: {doc_id}")
        print(f"   Document: {results['documents'][0][i]}")
        print(f"   Metadata: {results['metadatas'][0][i]}")
        print(f"   Cosine similarity (derived): {similarity:.4f}")
        print(f"   Cosine distance: {distance:.4f}")


if __name__ == "__main__":
    print("Total documents stored:", collection.count())

    # Semantic similarity search
    search("How can I find relevant information in a vector database?")

    # Metadata-filtered search
    search(
        "Detecting suspicious activity on a network",
        category="security"
    )

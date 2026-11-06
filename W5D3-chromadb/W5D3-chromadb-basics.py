
import chromadb

# Save the database locally so it persists between runs.
client = chromadb.PersistentClient(path="./chroma_data")

collection = client.get_or_create_collection(
    name="uzma_learning_docs",
    configuration={"hnsw": {"space": "cosine"}},
)

# 20 sample documents about AI and cybersecurity.
topics = [
    ("Machine learning learns patterns from data.", "AI"),
    ("Deep learning uses neural networks with multiple layers.", "AI"),
    ("Natural language processing works with human language.", "AI"),
    ("Computer vision helps machines interpret images.", "AI"),
    ("An embedding represents text as a numeric vector.", "AI"),
    ("ChromaDB stores vectors and supports similarity search.", "AI"),
    ("Retrieval augmented generation retrieves context for an LLM.", "AI"),
    ("Ollama can run language models on a local computer.", "AI"),
    ("Model evaluation measures performance on unseen data.", "AI"),
    ("Data preprocessing prepares data for machine learning.", "AI"),
    ("A firewall filters network traffic using security rules.", "Cybersecurity"),
    ("Intrusion detection identifies suspicious network activity.", "Cybersecurity"),
    ("Encryption protects information by making it unreadable without a key.", "Cybersecurity"),
    ("Multi-factor authentication adds another login verification step.", "Cybersecurity"),
    ("Phishing attacks trick people into revealing sensitive information.", "Cybersecurity"),
    ("Network monitoring helps identify unusual traffic patterns.", "Cybersecurity"),
    ("Access control determines who can use a resource.", "Cybersecurity"),
    ("A vulnerability is a weakness that can be exploited.", "Cybersecurity"),
    ("Incident response helps contain and investigate security events.", "Cybersecurity"),
    ("Hashing can help verify data integrity.", "Cybersecurity"),
]

# Use upsert so rerunning the script updates existing records.
collection.upsert(
    ids=[f"doc-{i:02}" for i in range(1, 21)],
    documents=[text for text, category in topics],
    metadatas=[
        {"category": category, "author": "Uzma"}
        for text, category in topics
    ],
)

print("Total documents:", collection.count())

# Search by meaning, not just exact keywords.
query = "How can I detect suspicious network activity?"
results = collection.query(
    query_texts=[query],
    n_results=3,
)

print("\nTop 3 similar documents:")
for document, metadata, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    print(f"\nCategory: {metadata['category']}")
    print(f"Text: {document}")
    print(f"Cosine distance: {distance:.4f}")

# Metadata filtering: search only cybersecurity documents.
filtered = collection.query(
    query_texts=[query],
    n_results=3,
    where={"category": "Cybersecurity"},
)

print("\nCybersecurity-only results:")
for document in filtered["documents"][0]:
    print("-", document)

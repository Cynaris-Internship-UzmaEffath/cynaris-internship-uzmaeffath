import csv
import time
from pathlib import Path
from statistics import mean

import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CHROMA_DIR = BASE_DIR / "chroma_db"

RESULTS_DIR.mkdir(exist_ok=True)

# Local Ollama configuration
Settings.embed_model = OllamaEmbedding(
    model_name="all-minilm:latest",
    base_url="http://localhost:11434",
)

Settings.llm = Ollama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    request_timeout=180.0,
)

Settings.chunk_size = 512
Settings.chunk_overlap = 50


# Ten questions for evaluating document retrieval
QUESTIONS = [
    "What is a network intrusion detection system?",
    "How does a network-based IDS monitor network traffic?",
    "Which network features can machine learning models use to detect attacks?",
    "What is a phishing attack?",
    "What are common warning signs of phishing?",
    "How can users prevent phishing attacks?",
    "What is Role-Based Access Control?",
    "What is the difference between authentication and authorization?",
    "How does multi-factor authentication improve security?",
    "Why is bcrypt used in password security?",
]


def get_source_names(response):
    names = []
    for source in response.source_nodes:
        name = source.node.metadata.get("file_name", "Unknown")
        if name not in names:
            names.append(name)
    return ", ".join(names)


def evaluate_index(index, backend_name):
    query_engine = index.as_query_engine(similarity_top_k=2)
    records = []

    print(f"\n{'=' * 65}")
    print(f"Evaluating: {backend_name}")
    print(f"{'=' * 65}")

    for number, question in enumerate(QUESTIONS, start=1):
        start_time = time.perf_counter()
        response = query_engine.query(question)
        elapsed = time.perf_counter() - start_time

        answer = str(response)
        sources = get_source_names(response)

        records.append({
            "backend": backend_name,
            "question": question,
            "answer": answer,
            "sources": sources,
            "latency_seconds": round(elapsed, 3),
        })

        print(f"\nQuestion {number}: {question}")
        print(f"Answer: {answer}")
        print(f"Sources: {sources}")
        print(f"Latency: {elapsed:.3f} seconds")

    return records


def main():
    print("LlamaIndex Document Indexing and Querying")
    print("-" * 65)

    # Load source documents
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".txt"],
    ).load_data()

    if not documents:
        raise ValueError("No text documents found in the data folder.")

    print(f"\nLoaded {len(documents)} documents.")

    # Part 1: In-memory vector index
    print("\nBuilding in-memory vector index...")
    start_time = time.perf_counter()

    memory_index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    memory_build_time = time.perf_counter() - start_time
    print(f"In-memory index built in {memory_build_time:.3f} seconds.")

    memory_results = evaluate_index(memory_index, "In-Memory")

    # Part 2: Persistent ChromaDB vector store
    print("\nConnecting to ChromaDB...")

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection_name = "cybersecurity_documents"

    # Clear previous records to avoid duplicate indexing on reruns
    try:
        old_collection = chroma_client.get_collection(collection_name)
        old_ids = old_collection.get()["ids"]

        if old_ids:
            old_collection.delete(ids=old_ids)

        print("Cleared previous ChromaDB records.")
    except Exception as error:
        if "does not exist" not in str(error).lower():
            raise

    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    print("\nBuilding ChromaDB-backed vector index...")
    start_time = time.perf_counter()

    chroma_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    chroma_build_time = time.perf_counter() - start_time
    print(f"ChromaDB index built in {chroma_build_time:.3f} seconds.")
    print(f"ChromaDB record count: {chroma_collection.count()}")

    chroma_results = evaluate_index(chroma_index, "ChromaDB")

    # Part 3: Save results
    all_results = memory_results + chroma_results

    output_file = RESULTS_DIR / "query_comparison.csv"

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "backend",
                "question",
                "answer",
                "sources",
                "latency_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    # Part 4: Compare average query latency
    memory_average = mean(
        record["latency_seconds"] for record in memory_results
    )
    chroma_average = mean(
        record["latency_seconds"] for record in chroma_results
    )

    print("\n" + "=" * 65)
    print("PERFORMANCE COMPARISON")
    print("=" * 65)
    print(f"In-memory index build time: {memory_build_time:.3f} seconds")
    print(f"ChromaDB index build time:   {chroma_build_time:.3f} seconds")
    print(f"In-memory average query:     {memory_average:.3f} seconds")
    print(f"ChromaDB average query:      {chroma_average:.3f} seconds")
    print(f"\nResults saved to: {output_file}")
    print("\nExecution completed successfully.")


if __name__ == "__main__":
    main()

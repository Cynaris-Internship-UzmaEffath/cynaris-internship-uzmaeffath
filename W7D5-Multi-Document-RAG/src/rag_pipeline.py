from pathlib import Path

import chromadb
import ollama
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "all-minilm:latest"


def load_documents():
    documents = []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8-sig")
        documents.append(
            Document(
                text=text,
                metadata={"file_name": file_path.name}
            )
        )

    if not documents:
        raise FileNotFoundError(f"No .txt documents found in {DATA_DIR}")

    print(f"Loaded {len(documents)} documents.")
    return documents


def build_index(documents):
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = chroma_client.get_or_create_collection(
        name="w7d5_documents"
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url="http://localhost:11434"
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )

    print("Document index created and stored in ChromaDB.")
    return index


def create_query_engine(index):
    llm = Ollama(
        model=LLM_MODEL,
        base_url="http://localhost:11434",
        request_timeout=180.0
    )

    return index.as_query_engine(
        llm=llm,
        similarity_top_k=3
    )


def main():
    print("Starting W7D5 Multi-Document RAG Pipeline")

    # Verify local Ollama service and models
    models = ollama.list()
    available_models = {
        model.model for model in models.models
    }

    if not any(LLM_MODEL in model for model in available_models):
        raise RuntimeError(f"Required LLM not found: {LLM_MODEL}")

    if not any(EMBED_MODEL in model for model in available_models):
        raise RuntimeError(f"Required embedding model not found: {EMBED_MODEL}")

    documents = load_documents()
    index = build_index(documents)
    query_engine = create_query_engine(index)

    print("\nRAG system is ready.")
    print("Enter 'exit' to stop.")

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("RAG session ended.")
            break

        if not question:
            continue

        response = query_engine.query(question)

        print("\nAnswer:")
        print(response)

        print("\nSource documents:")
        for source in response.source_nodes:
            file_name = source.node.metadata.get("file_name", "Unknown")
            print(f"- {file_name}")


if __name__ == "__main__":
    main()

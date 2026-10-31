
import argparse
import chromadb
import ollama

# Existing W5D3 database configuration
DB_PATH = "W5D3-chromadb/pdf_chroma_data"
COLLECTION_NAME = "pdf_knowledge"

EMBEDDING_MODEL = "all-minilm:latest"

SYSTEM_PROMPT = """
You are a helpful PDF-based question-answering assistant.

Answer using only the context provided.
Keep answers clear and concise.
Do not invent information.
If the answer is not available in the context,
clearly say that it was not found in the PDF.
"""

def retrieve_context(question, collection, top_k=2):
    """Retrieve relevant PDF chunks from ChromaDB."""

    embedding_response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question
    )

    query_embedding = embedding_response["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for document, metadata in zip(documents, metadatas):
        page = metadata.get("page", "Unknown")
        context_parts.append(
            f"Page {page}: {document}"
        )

    return "\n\n".join(context_parts)


def generate_answer(question, context, model):
    """Generate an answer using the selected local LLM."""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"PDF Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )

    return response["message"]["content"]


def main():
    parser = argparse.ArgumentParser(
        description="Local PDF Q&A Bot using Ollama and ChromaDB"
    )

    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        choices=["llama3.2:3b", "qwen2.5:3b"]
    )

    parser.add_argument(
        "--question",
        default="What is the main topic of this PDF?"
    )

    args = parser.parse_args()

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        raise RuntimeError("The ChromaDB collection is empty.")

    print("Local PDF Q&A Bot")
    print("-------------------------")
    print("Model:", args.model)
    print("Stored vectors:", collection.count())

    context = retrieve_context(
        args.question,
        collection
    )

    if not context.strip():
        print("No relevant PDF context found.")
        return

    print("\nRetrieved Context:")
    print(context[:1000])

    answer = generate_answer(
        args.question,
        context,
        args.model
    )

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()

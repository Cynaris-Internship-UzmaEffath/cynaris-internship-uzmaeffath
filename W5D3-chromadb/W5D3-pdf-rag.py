
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "sample.pdf"
DB_PATH = BASE_DIR / "pdf_chroma_data"

EMBED_MODEL = "all-minilm"
CHAT_MODEL = "qwen2.5:3b"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
TOP_K = 3


def extract_pdf_text(pdf_path):
    """Extract readable text from each PDF page."""
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append((page_number, text))

    if not pages:
        raise ValueError(
            "No readable text found. Try a text-based PDF."
        )

    return reader, pages


def split_into_chunks(pages):
    """Split page text into overlapping word chunks."""
    chunks = []
    metadatas = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page_number, text in pages:
        words = text.split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + CHUNK_SIZE]

            if not chunk_words:
                continue

            chunks.append(" ".join(chunk_words))
            metadatas.append({
                "page": page_number,
                "source": PDF_PATH.name,
            })

    return chunks, metadatas


def main():
    # 1. Check the PDF path.
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Place sample.pdf beside this Python script."
        )

    print("PDF:", PDF_PATH)
    print("Checking Ollama models...")

    # 2. Extract PDF text.
    reader, pages = extract_pdf_text(PDF_PATH)
    print("PDF pages:", len(reader.pages))
    print("Pages containing text:", len(pages))

    # 3. Create chunks.
    chunks, metadatas = split_into_chunks(pages)

    if not chunks:
        raise ValueError("No text chunks were created.")

    print("Chunks created:", len(chunks))

    # 4. Generate local embeddings with Ollama.
    print("\nGenerating embeddings with Ollama...")
    embedding_response = ollama.embed(
        model=EMBED_MODEL,
        input=chunks,
    )
    embeddings = embedding_response["embeddings"]

    # 5. Create a persistent ChromaDB collection.
    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_or_create_collection(
        name="pdf_knowledge",
        configuration={"hnsw": {"space": "cosine"}},
    )

    # Clear old chunks so reruns use the current PDF.
    existing = collection.get(include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Store the chunks, embeddings and page metadata.
    collection.upsert(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("Chunks stored:", collection.count())
    print("ChromaDB location:", DB_PATH)

    # 6. Accept a question from the user.
    question = input(
        "\nAsk a question about the PDF: "
    ).strip()

    if not question:
        print("No question entered. Exiting.")
        return

    # 7. Embed the question with the SAME embedding model.
    query_response = ollama.embed(
        model=EMBED_MODEL,
        input=question,
    )
    query_embedding = query_response["embeddings"][0]

    # 8. Retrieve the three most relevant chunks.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, collection.count()),
    )

    documents = results["documents"][0]
    result_metadata = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n--- Retrieved PDF Chunks ---")

    for index, (document, metadata, distance) in enumerate(
        zip(documents, result_metadata, distances),
        start=1,
    ):
        print(
            f"\nChunk {index} | Page {metadata['page']} "
            f"| Cosine distance: {distance:.4f}"
        )
        print(document[:500])

    # 9. Build the context for the local language model.
    context = "\n\n".join(
        f"Page {metadata['page']}: {document}"
        for document, metadata in zip(
            documents, result_metadata
        )
    )

    # 10. Generate an answer grounded in retrieved text.
    print("\nGenerating answer with Ollama...")

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided "
                    "PDF context. Be clear and concise. "
                    "If the context does not contain the answer, "
                    "say that the answer was not found in the "
                    "retrieved text. Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PDF context:\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    )

    print("\n--- RAG Answer ---")
    print(response["message"]["content"])

    print("\nW5D3 PDF RAG completed successfully.")


if __name__ == "__main__":
    main()

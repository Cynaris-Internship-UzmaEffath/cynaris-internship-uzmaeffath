
import json
import urllib.request

import chromadb
from pypdf import PdfReader


PDF_PATH = r"W5D3-chromadb\sample.pdf"
OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "all-minilm:latest"
LLM_MODEL = "qwen2.5:3b"


def ollama_embeddings(texts):
    """Generate embeddings locally using Ollama."""
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


def extract_chunks(pdf_path, chunk_size=500, overlap=80):
    """Extract PDF text and split it into overlapping chunks."""
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def generate_answer(question, context):
    """Ask the local Ollama language model using retrieved context."""
    prompt = f"""Answer the question using only the context below.
If the context does not contain the answer, say that the PDF does not
provide enough information. Do not invent facts.

Context:
{context}

Question: {question}

Answer:"""

    payload = json.dumps({
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"].strip()


def main():
    chunks = extract_chunks(PDF_PATH)

    if not chunks:
        raise ValueError("No extractable text was found in the PDF.")

    print("PDF:", PDF_PATH)
    print("Chunks created:", len(chunks))

    client = chromadb.Client()
    collection = client.create_collection(
        name="w5d4_pdf_rag",
        metadata={"hnsw:space": "cosine"}
    )

    embeddings = ollama_embeddings(chunks)

    collection.add(
        ids=[f"chunk_{i + 1}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"chunk_number": i + 1}
            for i in range(len(chunks))
        ]
    )

    question = "How can a RAG system answer questions using a PDF?"
    query_embedding = ollama_embeddings([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(3, len(chunks)),
        include=["documents", "distances", "metadatas"]
    )

    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)

    print("\nQuestion:", question)
    print("\nTop retrieved chunks:")

    for i, chunk in enumerate(retrieved_chunks):
        distance = results["distances"][0][i]
        print(f"\n{i + 1}. Chunk {results['metadatas'][0][i]['chunk_number']}")
        print(f"Cosine distance: {distance:.4f}")
        print(chunk)

    print("\nGenerating answer with", LLM_MODEL, "...")
    answer = generate_answer(question, context)
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()

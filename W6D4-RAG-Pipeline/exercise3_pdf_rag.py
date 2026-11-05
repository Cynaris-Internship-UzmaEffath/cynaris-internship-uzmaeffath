from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM


def main():
    base_dir = Path(__file__).parent
    pdf_path = base_dir / "sample.pdf"
    db_path = base_dir / "pdf_chroma_db"

    # Step 1: Load PDF
    print("Loading PDF...")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    if not pages:
        print("ERROR: No text found in the PDF.")
        return

    print("Total pages loaded:", len(pages))

    # Step 2: Split PDF into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(pages)
    print("Total chunks created:", len(chunks))

    if not chunks:
        print("ERROR: No text chunks were created.")
        return

    # Step 3: Initialize local embedding model
    print("\nInitializing Ollama embeddings...")
    embeddings = OllamaEmbeddings(model="all-minilm:latest")

    # Step 4: Create ChromaDB vector store
    print("Creating ChromaDB vector store...")

    vector_store = Chroma(
        collection_name="pdf_rag",
        embedding_function=embeddings,
        persist_directory=str(db_path),
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Remove previous documents before indexing again
    existing_ids = vector_store._collection.get()["ids"]

    if existing_ids:
        vector_store.delete(ids=existing_ids)

    vector_store.add_documents(chunks)

    print("PDF chunks stored:", vector_store._collection.count())

    # Step 5: Initialize local LLM
    print("\nInitializing Ollama LLM...")
    llm = OllamaLLM(
        model="llama3.2:3b",
        temperature=0.2
    )

    # Step 6: Retrieve and answer questions
    questions = [
        "What is Retrieval-Augmented Generation (RAG)?",
        "What is the role of embeddings in a vector database?"
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print("QUESTION:", question)

        retrieved_docs = vector_store.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

        print("\nTOP 3 RETRIEVED CHUNKS:")

        for i, doc in enumerate(retrieved_docs, start=1):
            print(f"\nChunk {i}:")
            print(doc.page_content[:500])
            print("Source:", doc.metadata)

        prompt = f"""
You are a helpful assistant.
Answer the question using only the context below.
If the answer is not present in the context, say that the PDF
does not contain enough information.

Context:
{context}

Question:
{question}

Answer:
"""

        print("\nGENERATED ANSWER:")
        response = llm.invoke(prompt)
        print(response)


if __name__ == "__main__":
    main()
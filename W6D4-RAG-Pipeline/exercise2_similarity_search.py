
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def main():

    # Step 1: Initialize embedding model
    embeddings = OllamaEmbeddings(
        model="all-minilm:latest"
    )

    # Step 2: Connect to existing ChromaDB collection
    vector_store = Chroma(
        collection_name="internship_notes",
        embedding_function=embeddings,
        persist_directory="./W6D4-RAG-Pipeline/chroma_db"
    )

    print("Connected to ChromaDB successfully!")
    print("Total documents:", vector_store._collection.count())

    # Step 3: Define search query
    query = "How does gradient descent train a machine learning model?"

    # Step 4: Perform top-5 similarity search
    results = vector_store.similarity_search_with_score(
        query,
        k=5
    )

    print("\n===== TOP 5 SIMILARITY RESULTS =====")

    for i, (doc, distance) in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Document:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("Cosine distance:", round(distance, 4))

    # Step 5: Apply metadata filtering
    filtered_results = vector_store.similarity_search(
        query,
        k=5,
        filter={"source": "lecture"}
    )

    print("\n===== LECTURE DOCUMENTS ONLY =====")

    for i, doc in enumerate(filtered_results, start=1):
        print(f"\nResult {i}")
        print("Document:", doc.page_content)
        print("Metadata:", doc.metadata)

    # Step 6: Verify metadata filter
    if all(doc.metadata["source"] == "lecture"
           for doc in filtered_results):
        print("\nSUCCESS: Metadata filtering verified!")
    else:
        print("\nERROR: Unexpected metadata found.")


if __name__ == "__main__":
    main()

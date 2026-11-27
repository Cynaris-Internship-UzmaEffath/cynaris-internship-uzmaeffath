from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)

from load_documents import load_pdf_documents
from reader import ExtractiveReader


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_dense_pipeline():
    documents = load_pdf_documents()

    print("\nLoading document embedding model...")
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=MODEL_NAME
    )
    document_embedder.warm_up()

    embedded_documents = document_embedder.run(
        documents=documents
    )["documents"]

    document_store = InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )
    document_store.write_documents(embedded_documents)

    print("\nLoading query embedding model...")
    text_embedder = SentenceTransformersTextEmbedder(
        model=MODEL_NAME
    )
    text_embedder.warm_up()

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=3
    )

    reader = ExtractiveReader()

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("reader", reader)

    pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding"
    )

    pipeline.connect(
        "retriever.documents",
        "reader.documents"
    )

    print("\nDense Retrieval pipeline with Reader created successfully.")
    print(f"Documents indexed: {document_store.count_documents()}")

    return pipeline


if __name__ == "__main__":
    pipeline = build_dense_pipeline()

    query = "What is machine learning and how is it related to artificial intelligence?"

    print(f"\nQuery: {query}")

    results = pipeline.run({
        "text_embedder": {
            "text": query
        },
        "reader": {
            "query": query
        }
    })

    print("\n===== TOP 3 RETRIEVED DOCUMENTS =====")

    retrieved_documents = results["reader"]["documents"]

    for index, document in enumerate(retrieved_documents, start=1):
        print(f"\nResult {index}")
        print(f"Source: {document.meta['source']}")
        print(f"Similarity score: {document.score}")

    print("\n===== EXTRACTIVE READER OUTPUT =====")
    print(f"Answer: {results['reader']['answer']}")
    print(f"Source: {results['reader']['source']}")

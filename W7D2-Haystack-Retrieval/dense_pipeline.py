from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

from load_documents import load_documents


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_dense_pipeline():
    # Load PDF documents
    documents = load_documents()

    # Generate document embeddings
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=MODEL_NAME
    )
    document_embedder.warm_up()

    embedded_documents = document_embedder.run(
        documents=documents
    )["documents"]

    # Create document store
    document_store = InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )
    document_store.write_documents(embedded_documents)

    print(f"\nDocuments indexed: {document_store.count_documents()}")

    # Create dense retriever
    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=3
    )

    # Create query embedder
    text_embedder = SentenceTransformersTextEmbedder(
        model=MODEL_NAME
    )
    text_embedder.warm_up()

    # Build pipeline
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)

    pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding"
    )

    return pipeline


if __name__ == "__main__":
    pipeline = build_dense_pipeline()

    question = "What is machine learning and how is it related to artificial intelligence?"

    result = pipeline.run({
        "text_embedder": {
            "text": question
        }
    })

    print("\nQuestion:", question)
    print("\nTop 3 retrieved documents:")

    for index, document in enumerate(result["retriever"]["documents"], start=1):
        print(f"\nRank {index}")
        print("Source:", document.meta.get("source", "Unknown"))
        print("Content:", document.content[:300])


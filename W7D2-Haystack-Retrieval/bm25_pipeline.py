from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

from load_documents import load_documents


def build_bm25_pipeline():
    # Load PDF documents
    documents = load_documents()

    # Create an in-memory document store
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    print(f"\nDocuments indexed: {document_store.count_documents()}")

    # Create BM25 retriever
    retriever = InMemoryBM25Retriever(
        document_store=document_store,
        top_k=3
    )

    # Build pipeline
    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)

    return pipeline


if __name__ == "__main__":
    pipeline = build_bm25_pipeline()

    question = "What is machine learning and how is it related to artificial intelligence?"

    result = pipeline.run({
        "retriever": {
            "query": question
        }
    })

    print("\nQuestion:", question)
    print("\nTop 3 retrieved documents:")

    for index, document in enumerate(result["retriever"]["documents"], start=1):
        print(f"\nRank {index}")
        print("Source:", document.meta.get("source", "Unknown"))
        print("Content:", document.content[:300])

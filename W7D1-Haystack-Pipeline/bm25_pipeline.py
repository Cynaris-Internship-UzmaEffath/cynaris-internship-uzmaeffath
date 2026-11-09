from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

from load_documents import load_pdf_documents
from reader import ExtractiveReader


def build_bm25_pipeline():
    documents = load_pdf_documents()

    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    retriever = InMemoryBM25Retriever(
        document_store=document_store,
        top_k=3
    )

    reader = ExtractiveReader()

    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("reader", reader)

    pipeline.connect(
        "retriever.documents",
        "reader.documents"
    )

    print("\nBM25 pipeline with Reader created successfully.")
    print(f"Documents indexed: {document_store.count_documents()}")

    return pipeline


if __name__ == "__main__":
    pipeline = build_bm25_pipeline()

    query = "What is machine learning and how is it related to artificial intelligence?"

    print(f"\nQuery: {query}")

    results = pipeline.run({
        "retriever": {
            "query": query
        },
        "reader": {
            "query": query
        }
    })

    print("\n===== RETRIEVED DOCUMENTS =====")

    retrieved_documents = results["reader"]["documents"]

    for index, document in enumerate(retrieved_documents, start=1):
        print(f"\nResult {index}")
        print(f"Source: {document.meta['source']}")
        print(f"Score: {document.score}")

    print("\n===== EXTRACTIVE READER OUTPUT =====")
    print(f"Answer: {results['reader']['answer']}")
    print(f"Source: {results['reader']['source']}")


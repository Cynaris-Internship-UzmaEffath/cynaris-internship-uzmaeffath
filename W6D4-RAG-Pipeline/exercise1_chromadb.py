
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


def main():

    # Step 1: Initialize Ollama embedding model
    print("Initializing Ollama embeddings...")

    embeddings = OllamaEmbeddings(
        model="all-minilm:latest"
    )

    # Step 2: Prepare 20 documents
    texts = [
        "Machine learning enables computers to learn from data.",
        "Deep learning uses multiple neural network layers.",
        "Artificial intelligence helps machines perform intelligent tasks.",
        "Supervised learning uses labelled training data.",
        "Unsupervised learning discovers patterns in data.",
        "Classification predicts categories for input data.",
        "Regression predicts continuous numerical values.",
        "Decision trees make predictions using branching rules.",
        "Random forests combine multiple decision trees.",
        "Neural networks contain interconnected artificial neurons.",
        "Gradient descent minimizes a model's loss function.",
        "Overfitting occurs when a model memorizes training data.",
        "Regularization helps reduce overfitting.",
        "CNNs are commonly used for image recognition.",
        "RNNs process sequential information.",
        "LSTMs help learn long-term dependencies.",
        "Natural language processing works with human language.",
        "Embeddings represent text as numerical vectors.",
        "RAG retrieves relevant documents before generating answers.",
        "Vector databases support semantic similarity searches."
    ]

    # Step 3: Create documents with metadata
    documents = [
        Document(
            page_content=text,
            metadata={
                "source": "lecture" if i % 2 == 0 else "article",
                "topic": "machine_learning"
            }
        )
        for i, text in enumerate(texts)
    ]

    # Step 4: Initialize ChromaDB collection
    print("Creating ChromaDB collection...")

    vector_store = Chroma(
        collection_name="internship_notes",
        embedding_function=embeddings,
        persist_directory="./W6D4-RAG-Pipeline/chroma_db",
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Step 5: Insert documents
    ids = [f"doc_{i}" for i in range(20)]

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    # Step 6: Verify collection
    count = vector_store._collection.count()

    print("\n----- VERIFICATION -----")
    print("Total documents:", count)

    if count == 20:
        print("SUCCESS: All 20 documents stored!")
    else:
        print("WARNING: Expected 20 documents.")

    print("Collection name: internship_notes")


if __name__ == "__main__":
    main()

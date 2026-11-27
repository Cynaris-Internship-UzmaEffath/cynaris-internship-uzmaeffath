from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = Path("W7D2-Haystack-Retrieval/data/pdfs")
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "all-minilm:latest"
LLM_MODEL = "llama3.2:3b"
COLLECTION_NAME = "w8d2_ragas_evaluation"


def build_vector_store(chunk_size=500, chunk_overlap=100):
    documents = []

    for pdf_path in sorted(SOURCE_DIR.glob("*.pdf")):
        pages = PyPDFLoader(str(pdf_path)).load()

        for page in pages:
            page.metadata["source"] = pdf_path.name

        documents.extend(pages)

    if not documents:
        raise ValueError(f"No PDF documents found in {SOURCE_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR)
    )

    print(f"Loaded PDF pages: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Vector database: {CHROMA_DIR}")

    return vector_store


def create_rag_chain(vector_store, retrieval_k=3):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": retrieval_k}
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the supplied context.
If the answer is not available in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:"""
    )

    def answer_question(question):
        retrieved_docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

        response = (prompt | llm).invoke({
            "context": context,
            "question": question
        })

        return {
            "question": question,
            "answer": response.content,
            "contexts": [doc.page_content for doc in retrieved_docs],
            "sources": [
                doc.metadata.get("source", "unknown")
                for doc in retrieved_docs
            ]
        }

    return answer_question


if __name__ == "__main__":
    store = build_vector_store()
    rag = create_rag_chain(store)

    result = rag(
        "What enables machine learning systems to improve their performance?"
    )

    print("\nQuestion:", result["question"])
    print("\nAnswer:", result["answer"])
    print("\nRetrieved sources:", result["sources"])


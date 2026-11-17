import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "eval" / "evaluation_dataset.json"
CHROMA_DIR = BASE_DIR / "src" / "chroma_db"
OUTPUT_PATH = BASE_DIR / "eval" / "results" / "optimized_k5_controlled_answers.json"

EMBEDDING_MODEL = "all-minilm:latest"
LLM_MODEL = "llama3.2:3b"
COLLECTION_NAME = "w8d2_ragas_evaluation"
RETRIEVAL_K = 5


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # Exact same prompt used in the baseline experiment
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the supplied context.
If the answer is not available in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:"""
    )

    results = []

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        print(f"\nProcessing {item['id']} ({index}/{len(dataset)})")

        documents = retriever.invoke(question)
        contexts = [doc.page_content for doc in documents]

        sources = list(dict.fromkeys(
            doc.metadata.get("source", "unknown")
            for doc in documents
        ))

        response = (prompt | llm).invoke({
            "context": "\n\n".join(contexts),
            "question": question
        })

        results.append({
            "id": item["id"],
            "question": question,
            "answer": response.content,
            "reference": item["reference"],
            "retrieved_contexts": contexts,
            "sources": sources,
            "expected_source": item["source"]
        })

        print("Answer:", response.content)
        print("Retrieved chunks:", len(contexts))

    output = {
        "retrieval_k": RETRIEVAL_K,
        "chunk_size": 500,
        "chunk_overlap": 100,
        "prompt": "same_as_baseline",
        "results": results
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nControlled k=5 generation completed.")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()

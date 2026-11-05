# W6D4 - RAG Pipeline using LangChain and ChromaDB

## Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, and locally running Ollama models.

The pipeline stores documents as embeddings, retrieves relevant information through similarity search, and generates answers using a local language model.

## Technologies Used

- Python 3.11
- LangChain
- ChromaDB
- Ollama
- PyPDF
- all-minilm:latest (Embedding Model)
- llama3.2:3b (Language Model)

## Exercises Completed

### Exercise 1 – ChromaDB Vector Store Setup

- Created a ChromaDB collection named `internship_notes`.
- Created 20 sample machine learning documents.
- Generated embeddings using Ollama.
- Stored documents and metadata in ChromaDB.
- Verified that all 20 documents were stored successfully.

### Exercise 2 – Similarity Search and Metadata Filtering

- Connected to the existing ChromaDB collection.
- Performed top-5 similarity searches.
- Retrieved relevant documents based on a query.
- Applied metadata filtering to retrieve lecture documents only.
- Verified the metadata filtering results.

### Exercise 3 – PDF-based RAG Pipeline

- Loaded a PDF using PyPDFLoader.
- Split the PDF into smaller text chunks.
- Generated embeddings using Ollama.
- Stored 8 chunks in a ChromaDB collection.
- Retrieved the top 3 relevant chunks for each question.
- Passed retrieved context to the local LLM.
- Generated answers using Ollama.

## Project Structure

```text
W6D4-RAG-Pipeline/
│
├── exercise1_chromadb.py
├── exercise2_similarity_search.py
├── exercise3_pdf_rag.py
├── sample.pdf
├── README.md
├── chroma_db/
└── pdf_chroma_db/
```

## How to Run

Activate the virtual environment and ensure Ollama is running.

Run each exercise from the project root:

```powershell
python W6D4-RAG-Pipeline\exercise1_chromadb.py

python W6D4-RAG-Pipeline\exercise2_similarity_search.py

python W6D4-RAG-Pipeline\exercise3_pdf_rag.py
```

## Verification Results

- 20 documents successfully stored in the first ChromaDB collection.
- Top-5 similarity search completed successfully.
- Metadata filtering verified successfully.
- PDF loaded successfully with 2 pages.
- 8 text chunks created and stored.
- Top-3 retrieval completed for both test questions.
- Answers generated successfully using the local Ollama model.

## Conclusion

Successfully implemented a RAG pipeline using LangChain, ChromaDB, and Ollama. The exercises demonstrate document storage, semantic similarity search, metadata filtering, PDF processing, context retrieval, and local LLM-based answer generation without requiring external API keys.

# W7D5 - Multi-Document RAG System

## 1. Project Overview

The Multi-Document Retrieval-Augmented Generation (RAG) System is a local AI application that retrieves relevant information from multiple text documents and generates context-aware answers using a locally hosted Large Language Model (LLM).

The system uses LlamaIndex to manage document indexing and retrieval, Ollama to run embedding and language models locally, and ChromaDB for persistent vector storage.

## 2. Objectives

- Load and process multiple text documents.
- Generate vector embeddings locally.
- Store document embeddings in ChromaDB.
- Retrieve relevant document chunks based on user queries.
- Generate answers using a local LLM.
- Display the source documents used to generate answers.
- Build a RAG application without requiring external LLM API keys.

## 3. Documents Used

The system uses five sample text documents:

1. artificial_intelligence.txt
2. cybersecurity.txt
3. cloud_computing.txt
4. natural_language_processing.txt
5. machine_learning.txt

Each document contains information about its respective technical domain.

## 4. Technologies and Tools

| Technology | Purpose |
|---|---|
| Python 3.11 | Main programming language |
| LlamaIndex | Document indexing and retrieval |
| Ollama | Local model execution |
| Llama 3.2 3B | Answer generation |
| all-MiniLM | Text embedding generation |
| ChromaDB | Persistent vector database |

## 5. System Architecture

The application follows these steps:

1. Document Loading: Reads all `.txt` files from the data directory.
2. Embedding Generation: Converts document text into numerical vector representations using the local all-MiniLM embedding model through Ollama.
3. Vector Storage: Stores the embeddings and document information in a persistent ChromaDB collection.
4. Query Processing: Accepts natural-language questions from the user.
5. Similarity Retrieval: Retrieves the three most relevant document chunks.
6. Answer Generation: Uses the Llama 3.2 3B model to generate an answer based on the retrieved context.
7. Source Display: Displays the names of the source documents associated with the retrieved content.

## 6. Project Structure

```text
W7D5-Multi-Document-RAG/
|
|-- data/
|   |-- artificial_intelligence.txt
|   |-- cybersecurity.txt
|   |-- cloud_computing.txt
|   |-- natural_language_processing.txt
|   |-- machine_learning.txt
|
|-- src/
|   |-- rag_pipeline.py
|
|-- chroma_db/
|   |-- chroma.sqlite3
|   |-- [vector collection storage]
|
|-- requirements.txt
|-- README.md
```

## 7. Installation and Setup

### Prerequisites

- Python 3.11
- Ollama installed and running
- Required Ollama models downloaded

### Install Python Dependencies

Activate the project's virtual environment and run:

```powershell
pip install -r requirements.txt
```

### Download Local Models

```powershell
ollama pull llama3.2:3b
ollama pull all-minilm:latest
```

Ensure the Ollama service is running before starting the application.

## 8. How to Run

Execute the following command from the repository root:

```powershell
python "W7D5-Multi-Document-RAG\src\rag_pipeline.py"
```

The application loads the documents, creates the vector index, and starts an interactive question-answering session.

Enter a question at the prompt.

To terminate the application, enter:

```text
exit
```

## 9. Testing and Validation

The following initial tests were performed:

- Successfully loaded five text documents.
- Successfully generated document embeddings.
- Created the ChromaDB persistent database.
- Successfully initialized the LlamaIndex query engine.
- Submitted a test question about the differences between Artificial Intelligence and Machine Learning.
- Generated a relevant answer.
- Displayed the retrieved source documents:
  - artificial_intelligence.txt
  - machine_learning.txt
  - natural_language_processing.txt

## 10. Key Learning Outcomes

- Understanding Retrieval-Augmented Generation architecture.
- Loading and processing multiple documents.
- Generating and working with vector embeddings.
- Creating document indexes using LlamaIndex.
- Implementing semantic similarity-based retrieval.
- Using ChromaDB for persistent vector storage.
- Integrating a local LLM with a retrieval pipeline.
- Displaying source information to improve answer traceability.
- Building an AI-powered question-answering application locally.

## 11. Privacy and Local Execution

This project uses locally hosted Ollama models and a local ChromaDB database. It does not require OpenAI, Google, or Anthropic API keys for its RAG pipeline.

## 12. Conclusion

The W7D5 Multi-Document RAG System demonstrates how information from multiple documents can be indexed, retrieved, and used to generate relevant answers through a local LLM.

The implementation provides practical experience with document retrieval, vector databases, embeddings, and LLM integration. These concepts can be extended to knowledge assistants, technical document search systems, and domain-specific question-answering applications.

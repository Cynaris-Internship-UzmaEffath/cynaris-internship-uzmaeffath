# LlamaIndex Document Indexing and Querying

## Project Overview
This project demonstrates document indexing and Retrieval-Augmented Generation (RAG) using LlamaIndex and locally hosted Ollama models.

It loads cybersecurity text documents, generates embeddings, retrieves relevant information, and generates answers using a local LLM.

## Objectives
- Load and process text documents.
- Generate embeddings using Ollama.
- Build an in-memory vector index using LlamaIndex.
- Create a query engine and run 10 questions.
- Integrate ChromaDB as a persistent vector store.
- Compare query results and latency.
- Save results in CSV format.

## Technologies Used
| Technology | Purpose |
|---|---|
| Python 3.11 | Programming language |
| LlamaIndex | Document indexing and retrieval |
| Ollama | Local model execution |
| all-minilm:latest | Text embeddings |
| llama3.2:3b | Answer generation |
| ChromaDB | Persistent vector database |

All models run locally. No external LLM API key is required.

## Dataset
Three cybersecurity documents are used:
- network_intrusion_detection.txt
- phishing_attacks.txt
- access_control.txt

Topics include intrusion detection, phishing prevention, authentication, authorization, RBAC, and password security.

## Implementation
1. Load documents using SimpleDirectoryReader.
2. Generate embeddings using OllamaEmbedding.
3. Build an in-memory VectorStoreIndex.
4. Create a query engine and execute 10 questions.
5. Connect LlamaIndex to persistent ChromaDB.
6. Execute the same 10 questions using ChromaDB.
7. Record answers, source filenames, and latency in a CSV file.

## Results
The experiment compares in-memory indexing and ChromaDB using the same 10 questions.

| Metric | In-memory | ChromaDB |
|---|---:|---:|
| Index build time | 5.554 sec | 2.580 sec |
| Average query time | 12.505 sec | 4.238 sec |
| Number of queries | 10 | 10 |

These are observed results from this local run. Query latency includes retrieval and LLM answer generation.

## Output
The evaluation results are saved in:
results/query_comparison.csv

## How to Run
1. Activate the Python virtual environment.
2. Ensure Ollama is running and the required models are available.
3. Run:

python .\W7D3-LlamaIndex-Document-Querying\main.py

## Conclusion
This project demonstrates how LlamaIndex can be used to build a local RAG pipeline and how ChromaDB can provide persistent vector storage. The experiment compares both approaches using query latency and retrieved source information.

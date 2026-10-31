# W5D5 – Local Q&A Bot: Model Comparison

## Project Overview

This project implements a local PDF-based Question Answering system using Ollama and ChromaDB. The system retrieves relevant PDF content using semantic similarity and generates answers using locally running language models.

## Technologies Used

- Python
- Ollama
- ChromaDB
- Llama 3.2 (3B)
- Qwen 2.5 (3B)
- All-MiniLM embeddings

## Models Compared

| Feature               | Llama 3.2 (3B)   | Qwen 2.5 (3B) |
| --------------------- | ---------------- | ------------- |
| Model execution       | Local            | Local         |
| Response generation   | Successful       | Successful    |
| Context-based answers | Yes              | Yes           |
| Response style        | More explanatory | More concise  |

## Test Results

Both models were tested using the same questions and retrieved PDF context.

### Observations

1. Both models successfully explained how ChromaDB and Ollama work together in a RAG system.
2. Both correctly explained how intrusion detection helps identify suspicious network activity.
3. Llama generally provided slightly more explanatory answers, whereas Qwen produced shorter and more direct responses.

## RAG Evaluation

The system was also tested with a question about the invention of the telephone, which was not covered in the PDF.

Llama correctly identified that the information was unavailable in the provided context.

This demonstrates the importance of context-grounded generation in reducing unsupported answers.

## Conclusion

Both Llama 3.2 (3B) and Qwen 2.5 (3B) successfully worked with the local ChromaDB knowledge base.

This project demonstrates how local language models, embeddings and vector databases can be integrated to build a privacy-friendly PDF question-answering system without relying on external cloud LLM APIs.

## Future Improvements

- Add more PDF documents and improve knowledge-base coverage.
- Evaluate answer faithfulness and relevance using Ragas.
- Experiment with different retrieval settings.
- Track experiments using MLflow.

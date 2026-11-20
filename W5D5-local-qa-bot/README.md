# W5D5 – Local Q&A Bot using Ollama and ChromaDB

## Overview

This project implements a local PDF Question Answering system using Retrieval-Augmented Generation (RAG).

It uses ChromaDB to retrieve relevant PDF content and Ollama to generate answers using locally running language models.

## Technologies

- Python
- Ollama
- ChromaDB
- Llama 3.2 (3B)
- Qwen 2.5 (3B)
- All-MiniLM embeddings

## Project Files

| File                      | Description                           |
| ------------------------- | ------------------------------------- |
| `qa_bot.py`               | PDF-based question-answering pipeline |
| `compare_models.py`       | Compares Llama and Qwen               |
| `setup_ollama.py`         | Tests five prompts using Ollama       |
| `ollama_responses.json`   | Stores five prompt responses          |
| `comparison_results.json` | Stores model comparison results       |
| `comparison_report.md`    | Documents comparison observations     |
| `SELF_REVIEW.md`          | Project self-review checklist         |

## Execution

Activate the project virtual environment.

Run five-prompt inference:

```bash
python W5D5-local-qa-bot/setup_ollama.py
```

Run the Q&A bot:

```bash
python W5D5-local-qa-bot/qa_bot.py --model llama3.2:3b
```

Run with Qwen:

```bash
python W5D5-local-qa-bot/qa_bot.py --model qwen2.5:3b
```

Compare both models:

```bash
python W5D5-local-qa-bot/compare_models.py
```

## Key Learning Outcomes

- Running LLMs locally using Ollama.
- Generating embeddings using All-MiniLM.
- Retrieving relevant PDF content using ChromaDB.
- Implementing a local RAG pipeline.
- Comparing different local language models.
- Saving model outputs and documenting results.

## Conclusion

This project demonstrates how Ollama and ChromaDB can be integrated to build a local, privacy-friendly PDF question-answering assistant without depending on external cloud LLM APIs.

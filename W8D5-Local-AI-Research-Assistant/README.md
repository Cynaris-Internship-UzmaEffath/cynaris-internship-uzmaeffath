# W8D5 – Local AI Research Assistant

## Overview

A local AI-powered research assistant that generates structured technical research reports using CrewAI, LangGraph and Ollama. The project tracks execution details with MLflow and evaluates answer relevancy using Ragas.

The application runs locally and does not require external LLM API keys.

## Architecture

1. **CrewAI:** Creates and executes the research agent and task.
2. **Ollama:** Runs the `llama3.2:3b` language model locally.
3. **LangGraph:** Orchestrates the report-generation workflow.
4. **MLflow:** Tracks experiment parameters, execution time, report length and report artifacts using SQLite.
5. **Ragas:** Evaluates answer relevancy using a local LLM and embedding model.

## Workflow

Research Topic → CrewAI Research Agent → Ollama LLM → LangGraph Workflow → Markdown Report → MLflow Tracking → Ragas Evaluation

## Project Structure

```text
W8D5-Local-AI-Research-Assistant/
├── research_assistant/
│   ├── __init__.py
│   ├── agent.py
│   └── workflow.py
├── tests/
│   ├── test_agent.py
│   └── test_workflow.py
├── outputs/
│   ├── research_report.md
│   └── evaluation_results.json
├── main.py
├── mlflow_tracking.py
├── evaluate.py
├── README.md
└── .gitignore
```

## Technologies

- Python 3.11
- CrewAI
- LangGraph
- Ollama
- MLflow
- Ragas
- LangChain
- SQLite
- pytest

## Requirements

- Python 3.11
- Ollama installed and running
- `llama3.2:3b` model
- `all-minilm:latest` embedding model

## Run the Application

Ensure Ollama is running and the required models are available. From the project directory, execute:

```powershell
python .\main.py
```

The generated research report is saved to `outputs/research_report.md`.

Execution parameters, metrics and report artifacts are tracked using MLflow.

## Run Ragas Evaluation

Execute:

```powershell
python .\evaluate.py
```

The answer relevancy evaluation result is saved to `outputs/evaluation_results.json`.

The evaluation uses local Ollama models without requiring external API keys.

## Run Tests

Execute:

```powershell
python -m pytest .\tests -v
```

The test suite covers the research agent and LangGraph workflow.

## View MLflow Experiments

Start the MLflow tracking interface:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` in your browser to view experiment runs, parameters, metrics and logged artifacts.

## Privacy and Limitations

- LLM inference and embedding operations run locally through Ollama.
- No external LLM API keys are required.
- Answer relevancy measures response relevance; it does not independently verify factual accuracy.
- Research output quality depends on the local model and the supplied research topic.

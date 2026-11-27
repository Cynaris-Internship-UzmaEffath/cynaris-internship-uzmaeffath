# W9D5 - Automated Research Report Agent

## Overview
A local automated research report pipeline combining CrewAI, LangGraph and MLflow.

## Architecture
1. CrewAI Researcher gathers structured research information.
2. CrewAI Analyst evaluates findings.
3. CrewAI Writer generates a Markdown report.
4. LangGraph orchestrates the workflow.
5. MLflow tracks parameters, metrics and output artifacts.

## Technologies
- Python 3.11
- CrewAI
- LangGraph
- MLflow
- Custom deterministic local LLM

## Execution
Activate the virtual environment and run:

```bash
python research_agent.py

## Validation
- Automated test: `python -m unittest discover -s tests -v`
- Result: 1 test passed.
- Report generation, analysis, and evaluation checks completed successfully.
- Execution uses a deterministic local mock LLM; no Ollama or external API key is required.

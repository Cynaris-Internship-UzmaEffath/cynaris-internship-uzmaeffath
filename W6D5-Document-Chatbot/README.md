# W6D5 - LangChain Chains, Memory and Agents

## Overview

This project demonstrates three important LangChain concepts:

1. LLM chains using PromptTemplate and OutputParser.
2. Conversation memory using LangGraph persistence.
3. Agents using tools for calculations and information searches.

The project uses a locally running Ollama model, so external API keys are not required.

## Technologies Used

- Python 3.11
- LangChain
- LangGraph
- Ollama
- llama3.2:3b
- PowerShell

## Project Structure

```text
W6D5-Document-Chatbot/
|
|-- document_chatbot.py
|-- chain_output.txt
|
|-- chatbot_memory.py
|-- memory_output.txt
|
|-- chatbot_agent.py
|-- agent_output.txt
|
|-- README.md
```

## 1. LangChain Chain

Implemented a chain using:

PromptTemplate -> OllamaLLM -> StrOutputParser

Tested the chain with five different questions and saved the generated responses in `chain_output.txt`.

## 2. Conversation Memory

Implemented conversation memory using LangGraph's InMemorySaver checkpointer.

- Tested five conversation turns.
- Verified that the assistant remembered the user's name and learning topic.
- Confirmed that all 10 messages (5 user messages and 5 assistant responses) were retained.

Evidence: `memory_output.txt`

## 3. LangChain Agent and Tools

Created an agent using two tools:

- **web_search:** A simulated search tool that returns predefined information.
- **calculator:** A tool that evaluates basic mathematical expressions safely using Python AST operations.

Tested the agent with three tasks:

- Multiplication using the calculator.
- Information search about LangChain.
- Division using the calculator.

The agent successfully selected and executed the appropriate tools.

Evidence: `agent_output.txt`

## How to Run

Make sure Ollama is installed and running, and the `llama3.2:3b` model is available.

Activate the Python virtual environment and run each component separately:

```powershell
python document_chatbot.py
python chatbot_memory.py
python chatbot_agent.py
```

## Results

- Successfully implemented and tested an LLM chain.
- Verified conversation memory across five turns.
- Implemented an agent with two tools.
- Completed three agent test tasks successfully.
- Saved terminal outputs as test evidence.

## Note

The web search tool is simulated for this exercise. It does not perform live internet searches.

The conversation memory uses an in-memory checkpointer and is not intended for permanent storage across application restarts.

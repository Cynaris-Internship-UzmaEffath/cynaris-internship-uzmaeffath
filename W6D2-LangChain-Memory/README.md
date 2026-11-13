# W6D2 - LangChain Memory & Conversation History

## Overview

Implemented LangChain chains, conversation memory, and a two-tool AI agent using local Ollama models.

## Technologies Used

- Python 3.11.9
- LangChain 1.4.7
- LangChain Ollama 1.1.0
- LangChain Classic 1.0.8
- LangGraph
- Ollama
- Llama 3.2 (3B)

## Tasks Completed

### 1. Basic LangChain Chain

Implemented a chain using:

- PromptTemplate
- Ollama LLM
- StrOutputParser

Tested the chain with five different inputs.

### 2. Conversation Memory

Implemented ConversationBufferMemory to maintain conversation history across five turns.

Verified that the assistant could recall previously provided information.

### 3. Two-Tool Agent

Created an agent using LangChain's create_agent API.

Tools:

- Web Search Stub: Returns simulated search results.
- Calculator: Performs arithmetic calculations.

Tested the agent with three tasks.

### 4. Agent Conversation History

Integrated LangGraph InMemorySaver to maintain conversation history using a shared thread ID.

Verified tool execution and context retention across three tasks.

## Project Structure

```text
W6D2-LangChain-Memory/
│
├── chain.py
├── chain_with_memory.py
├── agent.py
└── README.md
```

## Execution

```powershell
python chain.py
python chain_with_memory.py
python agent.py
```

## Results

- Basic chain: 5 inputs tested successfully.
- Conversation memory: 5 turns verified.
- Agent: 3 tasks executed.
- Calculator: Sum of first 20 natural numbers = 210.
- Agent checkpointing: Conversation history maintained across tasks.

## Note

The web search tool is a simulated stub and does not perform live internet searches.

## Conclusion

Successfully implemented LangChain chains, conversation memory, and a two-tool agent using local Ollama inference.

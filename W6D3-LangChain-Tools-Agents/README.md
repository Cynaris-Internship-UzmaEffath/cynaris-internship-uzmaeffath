# W6D3 - LangChain Tools & Agents

## Objective

To understand LangChain chains, conversation memory, tools, and agents using a locally running Ollama model.

## Environment

- Python 3.11.9
- LangChain 1.4.7
- LangChain Ollama 1.1.0
- LangChain Classic 1.0.8
- Ollama model: llama3.2:3b
- Local inference using Ollama (no external API keys)

## Tasks Implemented

### 1. LangChain Chain – `chain.py`

- Created a prompt using `PromptTemplate`.
- Connected the prompt to `OllamaLLM`.
- Used `StrOutputParser` to generate text output.
- Tested with five topics: Machine Learning, Artificial Intelligence, Cybersecurity, Cloud Computing, and Deep Learning.

**Evidence:** `chain_output.txt`

### 2. Conversation Memory – `chain_with_memory.py`

- Implemented `ConversationBufferMemory`.
- Integrated memory with `ConversationChain`.
- Tested five conversation turns.
- Verified that the model could recall the user's name, course, and area of interest.

**Evidence:** `memory_output.txt`

**Note:** `ConversationBufferMemory` and `ConversationChain` are deprecated APIs in the installed LangChain version. They were retained for this exercise because the task specifically required ConversationBufferMemory.

### 3. LangChain Tools & Agent – `agent.py`

- Created a simulated web search tool using `@tool`.
- Created a calculator tool with basic arithmetic operations.
- Integrated both tools with a local `ChatOllama` model.
- Built an agent using `create_agent`.
- Successfully tested three tasks:
  - 23 × 57 = 1311
  - Search for information about LangChain agents.
  - 150 ÷ 5 = 30

**Evidence:** `agent_output.txt`

**Note:** The web search tool is simulated and does not retrieve live internet results.

## Project Structure

```text
W6D3-LangChain-Tools-Agents/
│
├── chain.py
├── chain_output.txt
├── chain_with_memory.py
├── memory_output.txt
├── agent.py
├── agent_output.txt
└── README.md
```

## How to Run

Activate the Python virtual environment and run:

```powershell
python chain.py
python chain_with_memory.py
python agent.py
```

## Self-Review Checklist

- [x] Implemented a LangChain chain using PromptTemplate, OllamaLLM, and StrOutputParser.
- [x] Tested the chain with five inputs.
- [x] Implemented ConversationBufferMemory.
- [x] Verified conversation history across five turns.
- [x] Created a simulated web search tool.
- [x] Created a calculator tool.
- [x] Integrated both tools with a LangChain agent.
- [x] Tested the agent with three tasks.
- [x] Saved execution outputs as evidence.
- [x] Used local Ollama inference without external API keys.
- [ ] Completed CIA Full Stack Mentor Mode code reviews (minimum two interactions).

## Conclusion

Successfully implemented and tested LangChain chains, conversation memory, tools, and agents using the local Ollama model llama3.2:3b. Saved execution outputs for all three tasks.

## Key Learnings

- Understood the difference between a LangChain chain and an agent.
- Learned how conversation memory maintains context across multiple turns.
- Understood how agents select and invoke tools based on user instructions.
- Explored local LLM integration using Ollama and ChatOllama.
- Learned to use LangChain's create_agent for tool-based workflows.

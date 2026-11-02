# W6D1: LangChain Fundamentals - Chains & Prompts

## Objective

Build and test a LangChain chain using PromptTemplate, Ollama and OutputParser. Implement conversation memory and a simple agent with two tools.

## Technologies Used

- Python 3.11.9
- LangChain 1.4.7
- LangChain Ollama 1.1.0
- LangChain Classic 1.0.8
- Ollama 0.40.13
- Llama 3.2 (3B)

## Project Structure

- `test_ollama.py` – Tests the local Ollama connection.
- `chain_demo.py` – Implements PromptTemplate, Ollama and StrOutputParser.
- `memory_demo.py` – Implements conversation memory and verifies five conversation turns.
- `agent_demo.py` – Implements an agent with Web Search Stub and Calculator tools.

## Practical Tasks Completed

### 1. LangChain Chain

Created a chain using PromptTemplate, Ollama LLM and StrOutputParser. Tested with five different questions.

### 2. Conversation Memory

Implemented ConversationBufferMemory and tested five conversation turns. Verified that the assistant retained the conversation history across all five turns, storing 10 messages.

### 3. Agent with Two Tools

Implemented a LangChain agent using:

- Web Search Stub: Returns simulated search results.
- Calculator: Evaluates basic arithmetic expressions safely.

Tested the agent with three tasks:

- Calculate (125 + 375) \* 2.
- Search for information about machine learning.
- Search for Python and calculate 25 \* 4.

### 4. Testing Results

- Ollama connection: Successful.
- Chain testing: Five inputs executed.
- Memory testing: Five turns verified.
- Agent testing: Three tasks executed successfully.
- Calculator results: 1000 and 100.

## How to Run

Activate the virtual environment and execute:

```powershell
python W6D1\test_ollama.py
python W6D1\chain_demo.py
python W6D1\memory_demo.py
python W6D1\agent_demo.py
```

## Limitations

- Web Search is a simulated stub and does not perform live internet searches.
- ConversationBufferMemory produces deprecation warnings in the installed LangChain version.
- LLM responses may occasionally contain factual inaccuracies and should be verified.

## Conclusion

Successfully implemented and tested LangChain chains, conversation memory and a two-tool agent using a locally hosted Ollama model.

## Viva Questions

**1. What is the difference between a chain and a single LLM call?**

A single LLM call sends a prompt directly to a language model and receives a response. A chain connects multiple components, such as a prompt template, LLM and output parser, into a sequence.

**2. What problem does conversation memory solve?**

Conversation memory stores previous messages so the assistant can use earlier information when responding to later questions. It helps maintain context across multiple turns.

**3. What is the ReAct pattern?**

ReAct stands for Reasoning and Acting. It allows an agent to reason about a task, select and use tools, observe their results and provide a final response.

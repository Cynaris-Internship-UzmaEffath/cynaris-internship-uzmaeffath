# W9D4 - Hybrid Agent Systems

## Objective
Demonstrate LangChain prompt chaining, conversation history, and tool integration without using Ollama or external API keys.

## Implementation

### 1. LangChain Chain
PromptTemplate -> RunnableLambda (mock LLM) -> StrOutputParser

Five questions are processed sequentially.

### 2. Conversation Memory
Conversation history is maintained in a Python list. Each turn stores the user question and assistant response. The fifth question uses the previous conversation context.

### 3. Tools
- web_search: Simulated search results; no live internet access.
- calculator: Evaluates basic arithmetic expressions using Python AST parsing.

### 4. Agent Demonstration
Three predefined tasks demonstrate tool selection, tool execution, observation, and final output.

Note: Tool selection is deterministic and predefined. This is a ReAct-style tool-use demonstration, not an autonomous LLM-driven ReAct agent.

## Technologies
- Python 3.11
- LangChain Core
- PromptTemplate
- RunnableLambda
- StrOutputParser
- LangChain Tool
- Python AST and JSON

## Execution
```powershell
python W9D4-Hybrid-Agent-Systems/w9d4_hybrid.py

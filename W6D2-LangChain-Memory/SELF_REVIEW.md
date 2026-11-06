# W6D2 – Self Review Checklist

## 1. LangChain Basic Chain

- [x] Implemented PromptTemplate.
- [x] Integrated local Ollama LLM using Llama 3.2 (3B).
- [x] Added StrOutputParser.
- [x] Tested five different inputs successfully.

## 2. Conversation Memory

- [x] Implemented ConversationBufferMemory.
- [x] Maintained conversation history across five turns.
- [x] Verified that the assistant recalls previously provided information.
- [x] Documented the deprecation warning from the installed LangChain version.

## 3. LangChain Agent

- [x] Implemented an agent using create_agent.
- [x] Added web_search simulated search tool.
- [x] Added calculator tool with restricted AST-based arithmetic evaluation.
- [x] Tested three agent tasks successfully.
- [x] Verified calculator output: 210.
- [x] Integrated LangGraph InMemorySaver.
- [x] Verified conversation history across tasks using a shared thread ID.

## 4. Code Quality and Validation

- [x] Python syntax compilation successful.
- [x] Local Ollama inference used without external API keys.
- [x] README documentation added.
- [x] CIA Full Stack Mentor Mode review – two interactions completed.

## 5. Final Submission

- [ ] Two descriptive Git commits completed.
- [ ] Changes pushed to the W6 branch.
- [ ] GitHub pull request created.
- [ ] Output evidence captured and submitted.

## Notes

The web_search tool returns simulated results and does not perform live internet searches.

ConversationBufferMemory produces a deprecation warning in the installed LangChain version, but the implementation successfully demonstrates the memory functionality required by this assignment.

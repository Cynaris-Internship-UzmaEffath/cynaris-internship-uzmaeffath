from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOllama(model="llama3.2:3b", temperature=0)

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "You are a helpful AI assistant. "
        "Use the conversation history to answer questions accurately."
    ),
    checkpointer=checkpointer
)

if __name__ == "__main__":
    session_id = "w6d5-session"

    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    questions = [
        "My name is Alex.",
        "I am learning LangChain.",
        "What is my name?",
        "What framework am I learning?",
        "Summarize what you know about me from this conversation."
    ]

    print("===== W6D5: LangGraph Conversation Memory Testing =====")

    for index, question in enumerate(questions, start=1):
        print(f"\nTurn {index}")
        print("User:", question)

        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config
        )

        print("Assistant:", response["messages"][-1].content)

    final_state = agent.get_state(config)
    total_messages = len(final_state.values["messages"])

    print("\n===== MEMORY VERIFICATION =====")
    print("Total messages:", total_messages)
    print("Expected messages: 10")
    print(
        "Memory test:",
        "PASSED" if total_messages == 10 else "FAILED"
    )
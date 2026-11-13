
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver


# 1. Web Search Stub
@tool
def web_search(query: str) -> str:
    """Search for information using a simulated web search."""
    return (
        f"Simulated search result for '{query}': "
        "LangChain is a framework for building LLM applications. "
        "Ollama allows developers to run language models locally."
    )


# 2. Calculator Tool
@tool
def calculator(expression: str) -> str:
    """Calculate basic arithmetic expressions."""
    import ast
    import operator

    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def calculate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](
                calculate(node.left),
                calculate(node.right)
            )

        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = calculate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value

        raise ValueError("Unsupported mathematical expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(calculate(tree.body))
    except Exception as error:
        return f"Calculation error: {error}"


# 3. Initialize local Ollama model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# 4. Initialize conversation memory
checkpointer = InMemorySaver()


# 5. Create agent with two tools and memory
agent = create_agent(
    model=llm,
    tools=[web_search, calculator],
    system_prompt=(
        "You are a helpful AI assistant. "
        "You have access to web_search and calculator tools. "
        "When the user explicitly requests web_search, you MUST call web_search. "
        "When the user explicitly requests calculator, you MUST call calculator. "
        "Use previous conversation messages when answering follow-up questions. "
        "Never claim a simulated search result is real-time information."
    ),
    checkpointer=checkpointer
)


# 6. Run three tasks in the same conversation
if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "w6d2-conversation"
        }
    }

    tasks = [
        "Use web_search to look up information about LangChain.",

        "Use calculator to calculate the sum of the first 20 natural numbers.",

        "Use web_search to find information about Ollama, "
        "then explain how it relates to LangChain and the previous calculation."
    ]

    for index, task in enumerate(tasks, start=1):

        print(f"\n===== TASK {index} =====")
        print(f"User: {task}")

        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": task}
                ]
            },
            config=config
        )

        print("\nAgent response:")

        for message in result["messages"]:
            if message.type == "tool":
                print(f"[TOOL USED: {message.name}] {message.content}")

        print(result["messages"][-1].content)


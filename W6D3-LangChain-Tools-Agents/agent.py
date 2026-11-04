import ast
import operator

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent


# Initialize local Ollama chat model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# Simulated web search tool
@tool
def web_search(query: str) -> str:
    """Search for information using a simulated web search."""
    return f"Simulated search results for: {query}"


# Safe calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate basic arithmetic expressions."""

    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def calculate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](
                calculate(node.left),
                calculate(node.right)
            )

        if isinstance(node, ast.UnaryOp) and type(node.op) in operations:
            return operations[type(node.op)](calculate(node.operand))

        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(calculate(tree.body))
    except Exception:
        return "Invalid mathematical expression."


# Create agent with two tools
agent = create_agent(
    model=llm,
    tools=[web_search, calculator],
    system_prompt=(
        "You are a helpful assistant. "
        "Use the calculator tool for arithmetic questions. "
        "Use the web_search tool for search questions. "
        "After receiving the tool result, provide a concise final answer."
    )
)


# Test with three tasks
tasks = [
    "Calculate 23 multiplied by 57 using the calculator.",
    "Search for information about LangChain agents.",
    "Calculate 150 divided by 5 using the calculator."
]


for task in tasks:
    print("\n" + "=" * 50)
    print("TASK:", task)
    print("=" * 50)

    result = agent.invoke({
        "messages": [{"role": "user", "content": task}]
    })

    print("ANSWER:", result["messages"][-1].content)
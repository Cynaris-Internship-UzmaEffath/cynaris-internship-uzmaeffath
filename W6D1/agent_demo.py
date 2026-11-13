
import ast
import operator

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent


# Tool 1: Web Search Stub
@tool
def web_search(query: str) -> str:
    """Search for information using a simulated web search."""

    return (
        f"Simulated search result for: {query}. "
        "This is a demonstration stub, not a live web search."
    )


# Tool 2: Calculator
@tool
def calculator(expression: str) -> str:
    """Evaluate basic arithmetic expressions safely."""

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](
                evaluate(node.left),
                evaluate(node.right)
            )

        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](evaluate(node.operand))

        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(evaluate(tree.body))
    except Exception as error:
        return f"Calculation error: {error}"


# Initialize Ollama
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Register tools
tools = [web_search, calculator]

# Create modern LangChain agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant with two tools: web_search and calculator. "
        "Use the calculator for arithmetic questions. "
        "Use web_search for information searches. "
        "For tasks requiring both, use both tools. "
        "After receiving tool results, provide a concise final answer. "
        "Do not repeat a tool call if you already have its result. "
        "Clearly identify simulated search results as demonstrations."
    )
)

# Three practical tasks
tasks = [
    "Use calculator to calculate (125 + 375) * 2.",
    "Use web_search once to search for machine learning.",
    "Use web_search once to search for Python and calculator once to calculate 25 * 4."
]

# Execute tasks
for i, task in enumerate(tasks, 1):

    print("\n" + "=" * 50)
    print(f"AGENT TASK {i}")
    print("Task:", task)

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": 10}
        )

        print("\nFinal Answer:")
        print(result["messages"][-1].content)

        print("\nTools Used:")
        for message in result["messages"]:
            for tool_call in getattr(message, "tool_calls", []):
                print("-", tool_call["name"])

    except Exception as error:
        print("Task failed:", error)

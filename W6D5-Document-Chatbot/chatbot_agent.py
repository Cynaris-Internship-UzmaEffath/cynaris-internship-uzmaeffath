
import ast
import operator

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent


# 1. Initialize local Ollama model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# 2. Tool 1: Simulated web search
@tool
def web_search(query: str) -> str:
    """Search for general information using a simulated web search."""

    results = {
        "langchain": "LangChain is a framework for building applications powered by language models.",
        "rag": "RAG combines information retrieval with language model generation.",
        "python": "Python is a high-level programming language."
    }

    query_lower = query.lower()

    for keyword, information in results.items():
        if keyword in query_lower:
            return information

    return f"Simulated search results for: {query}"


# 3. Tool 2: Safe calculator
@tool
def calculator(expression: str) -> str:
    """Calculate basic arithmetic expressions."""

    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg
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
            return operations[type(node.op)](
                calculate(node.operand)
            )

        raise ValueError("Unsupported mathematical expression.")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(calculate(tree.body))

    except Exception:
        return "Invalid mathematical expression."


# 4. Create the agent with two tools
tools = [web_search, calculator]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful AI assistant. "
        "Use web_search for information searches. "
        "Use calculator for mathematical calculations. "
        "Always use the appropriate tool when required. "
        "After receiving the tool result, provide a concise answer."
    )
)


# 5. Test the agent with three tasks
if __name__ == "__main__":

    tasks = [
        "Calculate 256 multiplied by 37 using the calculator.",
        "Search for information about LangChain.",
        "Calculate 150 divided by 5 using the calculator."
    ]

    print("===== W6D5: LangChain Agent Testing =====")

    for index, task in enumerate(tasks, start=1):

        print(f"\nTask {index}")
        print("User:", task)

        result = agent.invoke({
            "messages": [
                {"role": "user", "content": task}
            ]
        })

        # Display tool execution and final response
        for message in result["messages"]:

            if getattr(message, "tool_calls", None):
                for call in message.tool_calls:
                    print("Tool selected:", call["name"])
                    print("Tool input:", call["args"])

            if message.type == "tool":
                print("Tool result:", message.content)

        print("Assistant:", result["messages"][-1].content)

    print("\n===== AGENT VERIFICATION =====")
    print("Registered tools:", [tool.name for tool in tools])
    print("Total tools:", len(tools))
    print("Agent testing completed.")

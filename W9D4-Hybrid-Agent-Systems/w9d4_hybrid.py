import ast
import operator
import json
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# 1. Deterministic mock LLM (no external model required)
def mock_llm(prompt):
    text = str(prompt)
    question = text.split("Question:")[-1].strip().splitlines()[0]
    q = question.lower()

    if "artificial intelligence" in q:
        return "Artificial intelligence enables machines to perform tasks that typically require human intelligence."
    if "25" in q and "16" in q:
        return "25 multiplied by 16 is 400."
    if "supervised learning" in q:
        return "Supervised learning trains a model using labelled examples."
    if "karnataka" in q:
        return "Bengaluru is the capital of Karnataka."
    if "discussed" in q or "earlier" in q:
        return "We discussed artificial intelligence, multiplication, supervised learning, and Karnataka."
    return "This is a deterministic demonstration response."

# 2. LangChain chain: PromptTemplate -> LLM -> OutputParser
prompt = PromptTemplate.from_template(
    "You are a helpful assistant.\n"
    "Conversation history: {history}\n"
    "Question: {question}\n"
    "Give a concise answer."
)

chain = prompt | RunnableLambda(mock_llm) | StrOutputParser()

# 3. Conversation memory across five turns
history = []
questions = [
    "What is artificial intelligence?",
    "What is 25 multiplied by 16?",
    "Explain supervised learning.",
    "What is the capital of Karnataka?",
    "Summarise what we discussed earlier."
]

chain_results = []
print("=== W9D4 CHAIN + MEMORY ===")

for number, question in enumerate(questions, 1):
    previous = "\n".join(
        f"User: {item['question']}\nAssistant: {item['answer']}"
        for item in history
    )
    answer = chain.invoke({"history": previous, "question": question})
    history.append({"question": question, "answer": answer})
    chain_results.append({"turn": number, "question": question, "answer": answer})
    print(f"\nTurn {number}: {question}")
    print("Answer:", answer)

print("\nMemory turns stored:", len(history))

# 4. Two LangChain tools
def web_search(query):
    return (
        f"SIMULATED SEARCH RESULT for '{query}': "
        "This is demonstration data, not live web information."
    )

def calculator(expression):
    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv
    }

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("Unsupported expression")

    return str(evaluate(ast.parse(expression, mode="eval")))

tools = {
    "web_search": Tool(
        name="web_search",
        func=web_search,
        description="Simulated web search for a topic."
    ),
    "calculator": Tool(
        name="calculator",
        func=calculator,
        description="Calculate a basic arithmetic expression."
    )
}

# 5. Simple deterministic ReAct-style tool selection
tasks = [
    {
        "task": "Find information about Bangalore population.",
        "tool": "web_search",
        "input": "Bangalore population"
    },
    {
        "task": "Calculate 123 multiplied by 456.",
        "tool": "calculator",
        "input": "123 * 456"
    },
    {
        "task": "Search for information about supervised learning.",
        "tool": "web_search",
        "input": "supervised learning"
    }
]

agent_results = []
print("\n=== W9D4 HYBRID AGENT ===")

for number, item in enumerate(tasks, 1):
    selected_tool = tools[item["tool"]]
    observation = selected_tool.invoke(item["input"])
    result = {
        "task_number": number,
        "task": item["task"],
        "action": item["tool"],
        "observation": observation,
        "final_answer": observation
    }
    agent_results.append(result)
    print(f"\nTask {number}: {item['task']}")
    print("Action:", item["tool"])
    print("Observation:", observation)
    print("Final answer:", observation)

# 6. Save evidence
(OUT / "chain_memory_results.json").write_text(
    json.dumps(chain_results, indent=2), encoding="utf-8"
)
(OUT / "agent_results.json").write_text(
    json.dumps(agent_results, indent=2), encoding="utf-8"
)

with (OUT / "run_output.txt").open("w", encoding="utf-8") as file:
    file.write("CHAIN + MEMORY\n")
    file.write(json.dumps(chain_results, indent=2))
    file.write("\n\nAGENT RESULTS\n")
    file.write(json.dumps(agent_results, indent=2))

print("\nEvidence saved in outputs/")
print("W9D4 execution completed.")

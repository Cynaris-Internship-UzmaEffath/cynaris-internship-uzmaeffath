
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferMemory

# 1. Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    output_key="output"
)

# 2. Create prompt template with conversation history
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a helpful AI assistant.

Previous conversation:
{history}

Current question: {input}

Answer:"""
)

# 3. Initialize Ollama
llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0.3
)

# 4. Initialize output parser
parser = StrOutputParser()

# 5. Build chain
chain = prompt | llm | parser


# 6. Run conversation with memory
def run_chain(user_input):

    history = memory.load_memory_variables({})["history"]

    response = chain.invoke({
        "history": history,
        "input": user_input
    })

    memory.save_context(
        {"input": user_input},
        {"output": response}
    )

    return response.strip()


# 7. Test five conversation turns
if __name__ == "__main__":

    questions = [
        "My name is Uzma.",
        "I am studying M.Tech in Computer Science.",
        "What is my name?",
        "What course am I studying?",
        "Summarize what you know about me from this conversation."
    ]

    print("\n===== LANGCHAIN CONVERSATION MEMORY =====\n")

    for index, question in enumerate(questions, start=1):

        print(f"Turn {index}")
        print(f"User: {question}")

        answer = run_chain(question)

        print(f"Assistant: {answer}")

        print("\nCurrent Conversation History:")
        print(memory.buffer)

        print("-" * 60)

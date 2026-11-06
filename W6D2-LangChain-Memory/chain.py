
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# 1. Create the prompt template
prompt = PromptTemplate.from_template(
    """You are a helpful AI assistant.
Answer the following question clearly and briefly.

Question: {input}

Answer:"""
)

# 2. Initialize the local Ollama model
llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0.3
)

# 3. Initialize the output parser
parser = StrOutputParser()

# 4. Build the LangChain chain
chain = prompt | llm | parser


# 5. Function to execute the chain
def run_chain(user_input):
    response = chain.invoke({"input": user_input})
    return response.strip()


# 6. Test with five different inputs
if __name__ == "__main__":

    questions = [
        "What is Artificial Intelligence?",
        "What is the capital of Karnataka?",
        "Explain supervised learning.",
        "What is cybersecurity?",
        "What is the difference between AI and ML?"
    ]

    print("\n===== LANGCHAIN BASIC CHAIN =====\n")

    for index, question in enumerate(questions, start=1):
        print(f"Test {index}")
        print(f"Question: {question}")

        answer = run_chain(question)

        print(f"Answer: {answer}")
        print("-" * 60)

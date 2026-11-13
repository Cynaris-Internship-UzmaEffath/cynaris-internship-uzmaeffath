
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# 1. Create the prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are a helpful AI assistant.
    Answer the following question clearly and simply.

    Question: {question}
    Answer:
    """
)

# 2. Initialize the local Ollama model
llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0
)

# 3. Initialize the output parser
parser = StrOutputParser()

# 4. Build the LangChain pipeline
chain = prompt | llm | parser


# 5. Test the chain with five inputs
if __name__ == "__main__":

    questions = [
        "What is LangChain?",
        "What is a Large Language Model?",
        "What is PromptTemplate?",
        "Why do we need an OutputParser?",
        "What is Retrieval Augmented Generation?"
    ]

    print("===== W6D5: LangChain Chain Testing =====")

    for index, question in enumerate(questions, start=1):
        print(f"\nTest {index}")
        print("Question:", question)

        response = chain.invoke({"question": question})

        print("Answer:", response)

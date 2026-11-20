
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# 1. Create Prompt Template
template = """
You are a helpful AI assistant.
Answer the following question accurately.
Stay focused on the topic asked.
Use simple English and keep the answer within 3 sentences.

Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# 2. Initialize Ollama LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# 3. Initialize Output Parser
output_parser = StrOutputParser()

# 4. Build LangChain
chain = prompt | llm | output_parser

# 5. Test with five inputs
questions = [
    "What is LangChain?",
    "What is machine learning?",
    "What is Python?",
    "What is cybersecurity?",
    "What is a neural network?"
]

for i, question in enumerate(questions, 1):
    print(f"\n--- Test {i} ---")
    print("Question:", question)

    response = chain.invoke({"question": question})

    print("Answer:", response)

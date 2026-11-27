
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

# Initialize local Ollama model
llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0.3
)

# Create prompt template
prompt = PromptTemplate.from_template(
    "Explain {topic} in simple terms with an example."
)

# Initialize output parser
parser = StrOutputParser()

# Build LangChain pipeline
chain = prompt | llm | parser

# Test with five different inputs
topics = [
    "Machine Learning",
    "Artificial Intelligence",
    "Cybersecurity",
    "Cloud Computing",
    "Deep Learning"
]

for topic in topics:
    print("\n" + "=" * 50)
    print("TOPIC:", topic)
    print("=" * 50)

    result = chain.invoke({"topic": topic})
    print("OUTPUT:", result)

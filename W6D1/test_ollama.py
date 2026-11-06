from langchain_ollama import ChatOllama

# Connect to local Ollama model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Test the model
response = llm.invoke("Explain LangChain in two simple sentences.")

print("Ollama Connection Successful!")
print("Response:")
print(response.content)
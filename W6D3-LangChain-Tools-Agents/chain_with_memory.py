
from langchain_ollama import OllamaLLM
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory

# Initialize Ollama model
llm = OllamaLLM(
    model="llama3.2:3b",
    temperature=0.3
)

# Initialize conversation memory
memory = ConversationBufferMemory()

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Five conversation turns
questions = [
    "My name is Uzma.",
    "I am studying M.Tech Computer Science.",
    "I am interested in Artificial Intelligence.",
    "What is my name?",
    "What course am I studying and what is my interest?"
]

# Execute conversation
for i, question in enumerate(questions, start=1):
    print(f"\n--- Conversation Turn {i} ---")
    print("User:", question)

    response = conversation.predict(input=question)

    print("AI:", response)

# Display stored conversation history
print("\n========== STORED MEMORY ==========")
print(memory.load_memory_variables({})["history"])

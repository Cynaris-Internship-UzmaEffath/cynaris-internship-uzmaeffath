
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 2. Initialize Ollama
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# 3. Create conversation prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the conversation history to answer questions accurately."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# 4. Create chain
chain = prompt | llm | StrOutputParser()

# 5. Function to maintain conversation
def chat(question):
    history = memory.load_memory_variables({})

    response = chain.invoke({
        "question": question,
        "chat_history": history["chat_history"]
    })

    memory.save_context(
        {"input": question},
        {"output": response}
    )

    return response

# 6. Test five conversation turns
questions = [
    "My name is Uzma.",
    "I am studying M.Tech in Computer Science.",
    "My favourite domain is Artificial Intelligence.",
    "What is my name?",
    "Which degree am I pursuing and what is my favourite domain?"
]

for i, question in enumerate(questions, 1):
    print(f"\n--- Turn {i} ---")
    print("User:", question)
    print("Assistant:", chat(question))

# 7. Verify conversation history
print("\n--- Conversation Memory Verification ---")
print("Total conversation messages:", len(memory.chat_memory.messages))
print(memory.chat_memory.messages)

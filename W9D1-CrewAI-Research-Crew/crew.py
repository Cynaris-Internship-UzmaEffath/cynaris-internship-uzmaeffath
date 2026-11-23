from crewai import Agent, LLM
from tools import DuckDuckGoSearchTool

# Local Ollama model
llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.2,
)

# Web search tool
web_search = DuckDuckGoSearchTool()

# Agent 1: Researcher with web search
researcher = Agent(
    role="Research Analyst",
    goal="Research topics using reliable web sources and provide evidence-based findings.",
    backstory=(
        "You are a detail-oriented research analyst. "
        "You search the web, examine source information, "
        "and distinguish verified evidence from unsupported claims."
    ),
    tools=[web_search],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 2: Writer
writer = Agent(
    role="Technical Writer",
    goal="Create a clear, structured report using research findings.",
    backstory=(
        "You are an experienced technical writer who converts "
        "complex research into readable and informative reports."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# Agent 3: Reviewer
reviewer = Agent(
    role="Quality Reviewer",
    goal="Review the report for accuracy, completeness, and readability.",
    backstory=(
        "You are a meticulous reviewer who checks factual consistency, "
        "identifies unsupported claims, and improves document quality."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

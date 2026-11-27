
from crewai import Agent, LLM

from tools import DuckDuckGoSearchTool
from code_execution_tool import CodeExecutionTool


# --------------------------------------------------
# 1. Local Ollama LLM
# --------------------------------------------------

llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.2,
)


# --------------------------------------------------
# 2. Initialize CrewAI tools
# --------------------------------------------------

web_search = DuckDuckGoSearchTool()
code_execution = CodeExecutionTool()


# --------------------------------------------------
# 3. Agent 1: Researcher
# --------------------------------------------------

researcher = Agent(
    role="Research Analyst",
    goal=(
        "Research topics using reliable web sources, "
        "analyze evidence, and provide accurate findings."
    ),
    backstory=(
        "You are a detail-oriented research analyst. "
        "You search the web, examine source information, "
        "perform numerical calculations using Python, "
        "and distinguish verified evidence from unsupported claims."
    ),
    tools=[
        web_search,
        code_execution,
    ],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# --------------------------------------------------
# 4. Agent 2: Writer
# --------------------------------------------------

writer = Agent(
    role="Technical Writer",
    goal=(
        "Create a clear, structured and informative report "
        "using the researcher's findings."
    ),
    backstory=(
        "You are an experienced technical writer who converts "
        "complex research into readable and informative reports. "
        "You use evidence carefully and avoid unsupported claims."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# --------------------------------------------------
# 5. Agent 3: Reviewer
# --------------------------------------------------

reviewer = Agent(
    role="Quality Reviewer",
    goal=(
        "Review the research report for factual accuracy, "
        "completeness, citation consistency and readability."
    ),
    backstory=(
        "You are a meticulous reviewer who checks factual "
        "consistency, identifies unsupported claims, verifies "
        "references and improves document quality."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

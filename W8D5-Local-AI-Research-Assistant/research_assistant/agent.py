
"""
CrewAI research agent powered by a local Ollama model.
"""

from crewai import Agent, Crew, LLM, Process, Task


# Configure the local language model.
local_llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.2,
)


def create_research_agent():
    """Create an AI agent for structured research assistance."""
    return Agent(
        role="AI Research Assistant",
        goal=(
            "Analyze research topics and produce clear, "
            "structured and informative research summaries."
        ),
        backstory=(
            "You are a careful research assistant who explains "
            "technical topics using accurate and understandable language."
        ),
        llm=local_llm,
        verbose=False,
        allow_delegation=False,
    )


def create_research_task(agent, topic):
    """Create a research task for the given topic."""
    return Task(
        description=(
            f"Analyze the following research topic: {topic}\n\n"
            "Prepare a structured report containing:\n"
            "1. Introduction\n"
            "2. Key concepts and technologies\n"
            "3. Important applications\n"
            "4. Current challenges\n"
            "5. Future research directions\n\n"
            "Use clear technical language. Do not invent citations, "
            "paper titles, authors or experimental results."
        ),
        expected_output=(
            "A well-organized research report with all five sections."
        ),
        agent=agent,
    )


def run_research(topic):
    """Run the CrewAI research assistant for a given topic."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Research topic cannot be empty.")

    agent = create_research_agent()
    task = create_research_task(agent, topic.strip())

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    research_topic = "Applications of artificial intelligence in cybersecurity"
    report = run_research(research_topic)
    print("\n===== RESEARCH REPORT =====\n")
    print(report)

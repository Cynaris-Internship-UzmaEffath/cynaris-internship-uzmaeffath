from crewai import Agent, Task, Crew, Process
from local_llm import DeterministicLocalLLM


def run_crew(topic):
    llm = DeterministicLocalLLM()

    researcher = Agent(
        role="Researcher",
        goal="Collect relevant background information",
        backstory="You are a careful research assistant.",
        llm=llm,
        verbose=False,
        max_iter=2,
        allow_delegation=False
    )

    analyst = Agent(
        role="Research Analyst",
        goal="Analyse research findings and identify insights",
        backstory="You analyse information objectively.",
        llm=llm,
        verbose=False,
        max_iter=2,
        allow_delegation=False
    )

    writer = Agent(
        role="Report Writer",
        goal="Prepare a clear research report",
        backstory="You write structured technical reports.",
        llm=llm,
        verbose=False,
        max_iter=2,
        allow_delegation=False
    )

    research_task = Task(
        description="Research the topic: {topic}. Identify applications, benefits, and challenges.",
        expected_output="Research findings and key points.",
        agent=researcher
    )

    analysis_task = Task(
        description="Analyse the research findings for {topic}. Identify benefits, risks, and practical considerations.",
        expected_output="A concise analysis of the findings.",
        agent=analyst
    )

    writing_task = Task(
        description="Write a structured report on {topic}, using the preceding research and analysis. Include an introduction, findings, analysis, and conclusion.",
        expected_output="A complete Markdown research report.",
        agent=writer,
        markdown=True
    )

    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=False
    )

    return crew.kickoff(inputs={"topic": topic})


if __name__ == "__main__":
    result = run_crew("Artificial Intelligence in Healthcare")
    print(result)

import json
from pathlib import Path

from crewai import Agent, Task, Crew, Process
from crewai.llms.base_llm import BaseLLM


topic = "AI-Based Crop Disease Detection"

# Synthetic test fixtures, not real research sources.
sources = [
    {
        "title": "Sample Source A - CNN-Based Image Classification",
        "url": "https://example.com/sample-cnn",
        "evidence": (
            "TEST DATA: CNNs can be evaluated for image classification "
            "tasks using labelled image datasets."
        ),
    },
    {
        "title": "Sample Source B - Agricultural Image Analysis",
        "url": "https://example.com/sample-agriculture",
        "evidence": (
            "TEST DATA: Image quality and dataset diversity are "
            "important considerations in agricultural image analysis."
        ),
    },
]


class DeterministicTestLLM(BaseLLM):
    """Scripted local LLM for testing CrewAI orchestration only."""

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):
        role = getattr(from_agent, "role", "")

        if role == "Researcher":
            response = f"""
RESEARCH BRIEF
Topic: {topic}

The following evidence is synthetic test data, not verified research.

1. CNN-Based Image Classification
Evidence: {sources[0]['evidence']}
Source: {sources[0]['url']}

2. Agricultural Image Analysis
Evidence: {sources[1]['evidence']}
Source: {sources[1]['url']}

Research limitation: These fixtures do not establish real-world
model accuracy or clinical/agricultural effectiveness.
"""
        elif role == "Writer":
            response = f"""# {topic}

## Introduction
This report demonstrates a local multi-agent research workflow
using synthetic test fixtures.

## Findings

### 1. CNN-Based Image Classification
{sources[0]['evidence']}
Source: {sources[0]['url']}

### 2. Agricultural Image Analysis
{sources[1]['evidence']}
Source: {sources[1]['url']}

## Conclusion
The test fixtures illustrate how evidence can be organized into
a structured report. They do not establish real-world performance.

## References
- {sources[0]['title']}: {sources[0]['url']}
- {sources[1]['title']}: {sources[1]['url']}

**Disclaimer:** All evidence above is synthetic test data.
"""
        elif role == "Reviewer":
            response = """REVIEW RESULT: PASS

Checklist:
- Title present: PASS
- Findings present: PASS
- Conclusion present: PASS
- References present: PASS
- Synthetic-data limitation disclosed: PASS

Review scope: Scripted workflow demonstration.
This is not an independent factual or scientific validation.
"""
        else:
            response = "Unexpected agent role. No scripted response available."

        return "Final Answer: " + response.strip()


llm = DeterministicTestLLM(model="deterministic-test")

researcher = Agent(
    role="Researcher",
    goal="Organize the supplied test evidence into a research brief.",
    backstory="You organize evidence and clearly identify synthetic data.",
    llm=llm,
    verbose=False,
)

writer = Agent(
    role="Writer",
    goal="Create a structured report from the research brief.",
    backstory="You write clear reports without inventing factual claims.",
    llm=llm,
    verbose=False,
)

reviewer = Agent(
    role="Reviewer",
    goal="Review the report structure and disclosure of test-data limitations.",
    backstory="You check report completeness and transparent limitations.",
    llm=llm,
    verbose=False,
)

research_task = Task(
    description=(
        f"Prepare a research brief about {topic}. "
        f"Organize these supplied synthetic fixtures: {json.dumps(sources)}. "
        "Clearly label all evidence as synthetic test data."
    ),
    expected_output="A research brief containing both fixtures and their URLs.",
    agent=researcher,
)

writing_task = Task(
    description=(
        "Write a Markdown report using the research brief. "
        "Include an introduction, findings, conclusion, references, "
        "and a clear synthetic-data disclaimer."
    ),
    expected_output="A complete Markdown research report.",
    agent=writer,
    context=[research_task],
)

review_task = Task(
    description=(
        "Check that the report contains a title, findings, conclusion, "
        "references, and a synthetic-data disclaimer. "
        "Return a checklist and state the review scope."
    ),
    expected_output="A review result with a checklist.",
    agent=reviewer,
    context=[research_task, writing_task],
)

crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=False,
)

print("=== W9D3 CREWAI MULTI-AGENT PIPELINE ===")
crew_result = crew.kickoff()

research_output = research_task.output.raw
report_output = writing_task.output.raw
review_output = review_task.output.raw

output_dir = Path(__file__).parent / "outputs"
output_dir.mkdir(exist_ok=True)

(output_dir / "research_report.md").write_text(
    report_output,
    encoding="utf-8",
)

results = {
    "topic": topic,
    "execution": "CrewAI sequential process",
    "llm": "Deterministic local test LLM",
    "live_inference": False,
    "live_web_research": False,
    "synthetic_test_data": True,
    "research": research_output,
    "report": report_output,
    "review": review_output,
    "crew_result": crew_result.raw,
}

(output_dir / "pipeline_results.json").write_text(
    json.dumps(results, indent=4),
    encoding="utf-8",
)

print("\nResearcher: Completed")
print("Writer: Completed")
print("Reviewer: Completed")
print("\n=== REVIEW OUTPUT ===")
print(review_output)
print("\nReport saved: outputs/research_report.md")
print("Results saved: outputs/pipeline_results.json")

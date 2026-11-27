"""LangGraph workflow for the local AI research assistant."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from research_assistant.agent import run_research


class ResearchState(TypedDict):
    topic: str
    report: str


def generate_report(state: ResearchState) -> ResearchState:
    """Generate a research report using the CrewAI agent."""
    report = run_research(state["topic"])
    return {
        "topic": state["topic"],
        "report": report,
    }


def build_research_workflow():
    """Build and compile the LangGraph research workflow."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("generate_report", generate_report)
    workflow.add_edge(START, "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


def run_workflow(topic: str) -> ResearchState:
    """Execute the research workflow."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Research topic cannot be empty.")

    app = build_research_workflow()
    return app.invoke({
        "topic": topic.strip(),
        "report": "",
    })

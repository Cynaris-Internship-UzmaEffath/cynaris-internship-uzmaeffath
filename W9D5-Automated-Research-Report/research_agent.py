from typing import TypedDict
from pathlib import Path
import json

import mlflow
from langgraph.graph import StateGraph, START, END
from crew_pipeline import run_crew


class ResearchState(TypedDict):
    topic: str
    research: str
    analysis: str
    report: str


def conduct_research(state: ResearchState):
    result = run_crew(state["topic"])

    tasks = result.tasks_output

    research = tasks[0].raw if len(tasks) > 0 else ""
    analysis = tasks[1].raw if len(tasks) > 1 else ""
    report = result.raw

    return {
        "research": str(research),
        "analysis": str(analysis),
        "report": str(report)
    }


def write_report(state: ResearchState):
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "research_report.md"
    report_path.write_text(state["report"], encoding="utf-8")

    return {}


def build_workflow():
    graph = StateGraph(ResearchState)

    graph.add_node("crew_research", conduct_research)
    graph.add_node("save_report", write_report)

    graph.add_edge(START, "crew_research")
    graph.add_edge("crew_research", "save_report")
    graph.add_edge("save_report", END)

    return graph.compile()


def generate_report(topic: str):
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    workflow = build_workflow()

    with mlflow.start_run(run_name="W9D5_Research_Report"):
        mlflow.log_param("topic", topic)
        mlflow.log_param("workflow", "CrewAI + LangGraph + MLflow")
        mlflow.log_param("execution", "deterministic_local")

        result = workflow.invoke({
            "topic": topic,
            "research": "",
            "analysis": "",
            "report": ""
        })

        report = result["report"]

        scores = {
            "research_completed": int(bool(result["research"])),
            "analysis_completed": int(bool(result["analysis"])),
            "report_generated": int(bool(report)),
            "has_conclusion": int("conclusion" in report.lower())
        }

        evaluation_path = output_dir / "evaluation_results.json"
        evaluation_path.write_text(
            json.dumps(scores, indent=4),
            encoding="utf-8"
        )

        mlflow.log_artifact(str(output_dir / "research_report.md"))
        mlflow.log_artifact(str(evaluation_path))
        mlflow.log_metric("report_word_count", len(report.split()))
        mlflow.log_metric("quality_checks_passed", sum(scores.values()))
        mlflow.log_metric("pipeline_completed", 1)

    return report, scores


if __name__ == "__main__":
    topic = "Artificial Intelligence in Healthcare"
    report, scores = generate_report(topic)

    print("\n" + "=" * 60)
    print("AUTOMATED RESEARCH REPORT")
    print("=" * 60)
    print(report)

    print("\nQUALITY CHECK RESULTS")
    print(json.dumps(scores, indent=4))

"""Entry point for the local AI research assistant."""

import time
from pathlib import Path

from research_assistant.workflow import run_workflow
from mlflow_tracking import track_research_run


def main():
    topic = "Applications of artificial intelligence in cybersecurity"

    print("\nStarting Local AI Research Assistant...")
    print(f"Research topic: {topic}\n")

    start_time = time.perf_counter()
    result = run_workflow(topic)
    duration = time.perf_counter() - start_time

    report = result["report"]

    print("\n========== RESEARCH REPORT ==========\n")
    print(report)

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "research_report.md"
    output_file.write_text(
        f"# Research Report\n\n**Topic:** {topic}\n\n{report}\n",
        encoding="utf-8",
    )

    print(f"\nReport saved to: {output_file}")
    print(f"Research execution time: {duration:.2f} seconds")

    track_research_run(topic, report, duration)


if __name__ == "__main__":
    main()

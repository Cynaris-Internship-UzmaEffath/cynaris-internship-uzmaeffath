"""MLflow experiment tracking for the local research assistant."""

from pathlib import Path

import mlflow


BASE_DIR = Path(__file__).resolve().parent
TRACKING_DB = BASE_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{TRACKING_DB.as_posix()}"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Local-AI-Research-Assistant")


def track_research_run(topic: str, report: str, duration_seconds: float):
    """Log research execution details and output to MLflow."""

    with mlflow.start_run(run_name="CrewAI-Local-Research"):
        mlflow.log_param("research_topic", topic)
        mlflow.log_param("llm_model", "llama3.2:3b")
        mlflow.log_param("agent_framework", "CrewAI")
        mlflow.log_param("workflow_framework", "LangGraph")
        mlflow.log_param("inference_provider", "Ollama")

        mlflow.log_metric("research_duration_seconds", duration_seconds)
        mlflow.log_metric("report_length_characters", len(report))

        report_file = BASE_DIR / "outputs" / "research_report.md"
        report_file.parent.mkdir(exist_ok=True)
        report_file.write_text(report, encoding="utf-8")

        mlflow.log_artifact(str(report_file))

        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        print("Research details and report logged successfully.")

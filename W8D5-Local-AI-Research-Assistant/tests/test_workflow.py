from unittest.mock import patch

import pytest

from research_assistant.workflow import run_workflow


@patch("research_assistant.workflow.run_research")
def test_run_workflow_returns_report(mock_run_research):
    mock_run_research.return_value = "Sample research report"

    result = run_workflow("AI in cybersecurity")

    assert result["topic"] == "AI in cybersecurity"
    assert result["report"] == "Sample research report"
    mock_run_research.assert_called_once_with("AI in cybersecurity")


def test_run_workflow_rejects_empty_topic():
    with pytest.raises(ValueError, match="Research topic cannot be empty"):
        run_workflow("   ")


@patch("research_assistant.workflow.run_research")
def test_run_workflow_strips_topic_whitespace(mock_run_research):
    mock_run_research.return_value = "Sample report"

    result = run_workflow("  AI in cybersecurity  ")

    assert result["topic"] == "AI in cybersecurity"
    mock_run_research.assert_called_once_with("AI in cybersecurity")

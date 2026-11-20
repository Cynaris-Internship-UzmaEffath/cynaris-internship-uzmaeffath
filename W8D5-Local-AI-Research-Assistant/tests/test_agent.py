from unittest.mock import MagicMock, patch

import pytest

from research_assistant.agent import (
    create_research_agent,
    create_research_task,
    run_research,
)


def test_create_research_agent():
    agent = create_research_agent()

    assert agent.role == "AI Research Assistant"
    assert agent.allow_delegation is False


def test_create_research_task():
    agent = create_research_agent()
    task = create_research_task(agent, "AI in cybersecurity")

    assert task.agent == agent
    assert "AI in cybersecurity" in task.description
    assert task.expected_output


def test_run_research_rejects_empty_topic():
    with pytest.raises(ValueError, match="Research topic cannot be empty"):
        run_research("   ")


@patch("research_assistant.agent.Crew")
@patch("research_assistant.agent.create_research_task")
@patch("research_assistant.agent.create_research_agent")
def test_run_research_returns_report(
    mock_create_agent,
    mock_create_task,
    mock_crew_class,
):
    mock_create_agent.return_value = MagicMock()
    mock_create_task.return_value = MagicMock()

    mock_crew = MagicMock()
    mock_crew.kickoff.return_value = "Generated research report"
    mock_crew_class.return_value = mock_crew

    result = run_research("AI in cybersecurity")

    assert result == "Generated research report"
    mock_crew.kickoff.assert_called_once()

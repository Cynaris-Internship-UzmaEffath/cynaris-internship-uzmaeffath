# W9D3 - CrewAI Multi-Agent Research Pipeline

## Objective
Demonstrate a sequential multi-agent research and report-generation workflow using CrewAI.

## Agents
1. **Researcher** - Organizes supplied evidence into a research brief.
2. **Writer** - Produces a structured Markdown report.
3. **Reviewer** - Checks report completeness and synthetic-data disclosures.

## Workflow
Researcher -> Writer -> Reviewer

The pipeline uses CrewAI Process.sequential.

## Technology
- Python 3.11
- CrewAI
- Custom deterministic local test LLM
- JSON
- Markdown

## Execution
From the repository root, run:

`python ".\W9D3-Research-Crew-Multi-Agent-Pipeline\main.py"`

## Output Files
- `outputs/research_report.md` - Generated Markdown report.
- `outputs/pipeline_results.json` - Execution results, research brief, report, and review.

## Testing and Limitations
The three-agent CrewAI execution completed successfully.

This implementation uses a scripted deterministic LLM. It does not perform live LLM inference or live web research.

The example.com references and evidence are synthetic test fixtures, not verified research sources.

The reviewer checks report structure and disclosure requirements. It does not independently validate scientific claims.

## Status
- Local CrewAI orchestration test: successful.
- Live LLM inference: not implemented.
- Live web research: not implemented.

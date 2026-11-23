# W9D1 - CrewAI Research Crew

## Overview

A multi-agent research workflow built using CrewAI. Three specialized agents collaborate to research a topic, write a structured report, and review the final output.

## Agents

1. **Research Analyst** - researches the topic and gathers relevant web sources.
2. **Technical Writer** - prepares a structured Markdown report.
3. **Quality Reviewer** - reviews the report for accuracy, completeness, and readability.

## Workflow

The crew follows a sequential process:

Research Analyst ? Technical Writer ? Quality Reviewer

## Technologies Used

- Python 3.11
- CrewAI 1.15.41
- DuckDuckGo Search (ddgs 9.16.1)
- Ollama
- Llama 3.2 3B

## Project Structure

```text
W9D1-CrewAI-Research-Crew/
+-- crew.py
+-- tasks.py
+-- tools.py
+-- main.py
+-- requirements.txt
+-- README.md
+-- .gitignore
+-- outputs/
    +-- crew_generated_report.md
    +-- crop_disease_research_report.md
    +-- evidence_based_research_report.md
    +-- search_sources.md
    +-- web_search_research_report.md

## Setup and Installation

### 1. Activate the virtual environment

From the internship repository root:

```powershell
.\.venv\Scripts\Activate.ps1
# W9D1 - CrewAI Research Crew

## Overview

A multi-agent research workflow built using CrewAI. Three specialized agents collaborate to research a topic, write a structured report, and review the final output.

## Agents

1. **Research Analyst** - researches the topic and gathers relevant web sources.
2. **Technical Writer** - prepares a structured Markdown report.
3. **Quality Reviewer** - reviews the report for accuracy, completeness, and readability.

## Workflow

The crew follows a sequential process:

Research Analyst ? Technical Writer ? Quality Reviewer

## Technologies Used

- Python 3.11
- CrewAI 1.15.41
- DuckDuckGo Search (ddgs 9.16.1)
- Ollama
- Llama 3.2 3B

## Project Structure

```text
W9D1-CrewAI-Research-Crew/
+-- crew.py
+-- tasks.py
+-- tools.py
+-- main.py
+-- requirements.txt
+-- README.md
+-- .gitignore
+-- outputs/
    +-- crew_generated_report.md
    +-- crop_disease_research_report.md
    +-- evidence_based_research_report.md
    +-- search_sources.md
    +-- web_search_research_report.md
from pathlib import Path

content = """# W9D1 - CrewAI Research Crew

## Overview

A multi-agent research workflow built using CrewAI. Three specialized agents collaborate to research a topic, write a structured report, and review the final output.

## Agents

1. **Research Analyst** - researches the topic and gathers relevant web sources.
2. **Technical Writer** - prepares a structured Markdown report.
3. **Quality Reviewer** - reviews the report for accuracy, completeness, and readability.

## Workflow

Research Analyst ? Technical Writer ? Quality Reviewer

The crew uses CrewAI's sequential process.

## Technologies Used

- Python 3.11
- CrewAI 1.15.41
- DuckDuckGo Search (ddgs 9.16.1)
- Ollama
- Llama 3.2 3B

## Project Structure

```text
W9D1-CrewAI-Research-Crew/
+-- crew.py
+-- tasks.py
+-- tools.py
+-- main.py
+-- requirements.txt
+-- README.md
+-- .gitignore
+-- outputs/
    +-- crew_generated_report.md
    +-- crop_disease_research_report.md
    +-- evidence_based_research_report.md
    +-- search_sources.md
    +-- web_search_research_report.md

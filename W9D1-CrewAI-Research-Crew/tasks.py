from crewai import Task
from crew import researcher, writer, reviewer

research_task = Task(
    description=(
        "Research the topic: {topic}. "
        "Use ONLY the following pre-collected web search results as your "
        "source material:\n\n{sources}\n\n"
        "Analyze the source titles, URLs and descriptions. "
        "Summarize findings about CNN architectures, PlantVillage, "
        "agricultural applications, benefits and limitations. "
        "Do not claim that you opened or verified a webpage. "
        "Do not invent authors, paper titles, statistics or findings. "
        "If the supplied descriptions do not support a claim, identify "
        "that information as not established by the available sources. "
        "Use each source number only when its description supports the claim."
    ),
    expected_output=(
        "An evidence-based research brief containing key findings and "
        "a numbered list of the supplied source titles and exact URLs. "
        "Identify any information that cannot be established from the "
        "provided search results."
    ),
    agent=researcher,
)

write_task = Task(
    description=(
        "Write a structured Markdown report about {topic} using only "
        "the Researcher's research brief and supplied sources. "
        "Include introduction, technical methodology, benefits, "
        "limitations, applications and conclusion. "
        "Use numbered inline citations only when supported by the source "
        "descriptions. Include each relevant source URL exactly once. "
        "Do not invent statistics, citations or references. "
        "Do not repeat sources to increase the reference count."
    ),
    expected_output=(
        "A structured Markdown report with supported numbered citations "
        "and a concise reference list of unique URLs."
    ),
    agent=writer,
    context=[research_task],
)

review_task = Task(
    description=(
        "Review the report against the Researcher's brief and the original "
        "supplied sources. Check every citation and numerical claim. "
        "Remove unsupported numerical claims or clearly label them "
        "unverified. Remove duplicate references. Preserve exact source "
        "URLs and ensure every citation points to a listed source. "
        "Do not invent replacement sources or facts."
    ),
    expected_output=(
        "A corrected Markdown report with consistent citations, "
        "unique source URLs and unsupported claims removed or flagged."
    ),
    agent=reviewer,
    context=[research_task, write_task],
)

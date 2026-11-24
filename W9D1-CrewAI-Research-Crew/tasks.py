
from crewai import Task
from crew import researcher, writer, reviewer


research_task = Task(
    description=(
        "Research the topic: {topic}.\n\n"
        "INITIAL SEARCH RESULTS:\n{sources}\n\n"

        "You have access to DuckDuckGo Web Search and Safe Python "
        "Code Execution tools.\n\n"

        "Use web search to find relevant evidence about PlantVillage, "
        "CNN architectures, agricultural crop disease detection, "
        "benefits and limitations.\n\n"

        "SOURCE INTEGRITY RULES — MANDATORY:\n"
        "1. Never invent a paper, author, journal, dataset, statistic, "
        "publication date or URL.\n"
        "2. Use only source titles and URLs actually returned by the "
        "initial search results or your web-search tool.\n"
        "3. Do not create references from memory or general knowledge.\n"
        "4. A search-result description is not proof that you opened "
        "or read the full webpage.\n"
        "5. Do not report accuracy percentages, image counts or other "
        "numbers unless they appear explicitly in a search result.\n"
        "6. If evidence is insufficient, write 'Not established by "
        "the available search results'.\n"
        "7. Use the Python tool only when an actual calculation is "
        "needed and numerical inputs are available. Never invent "
        "inputs or calculation results.\n\n"

        "Prepare a research brief with these sections:\n"
        "A. Supported findings\n"
        "B. CNN architectures identified in the sources\n"
        "C. Dataset information explicitly supported by sources\n"
        "D. Benefits and limitations supported by sources\n"
        "E. Calculations actually performed, if any\n"
        "F. Source ledger\n\n"

        "In the source ledger, list every source used with its exact "
        "title, exact URL and the specific claim its description "
        "supports. Include only sources actually returned by a tool."
    ),
    expected_output=(
        "An evidence-based research brief with supported findings "
        "and a source ledger containing exact titles and URLs. "
        "No fabricated references or unsupported statistics."
    ),
    agent=researcher,
)


write_task = Task(
    description=(
        "Write a structured Markdown report about {topic}.\n\n"

        "INITIAL SEARCH RESULTS:\n{sources}\n\n"

        "Use the Research Analyst's brief and source ledger. "
        "You may use only facts and references supported by that "
        "brief or the initial search results.\n\n"

        "MANDATORY RULES:\n"
        "1. Never invent references, paper titles, authors, journals, "
        "datasets, statistics or URLs.\n"
        "2. Do not use citations unless the exact source is present "
        "in the supplied evidence.\n"
        "3. Every reference must reproduce an exact supplied URL.\n"
        "4. Do not cite a source for a claim its description does "
        "not support.\n"
        "5. Do not turn general knowledge into a research finding.\n"
        "6. If evidence is missing, state that it is not established "
        "by the available search results.\n\n"

        "Use these sections:\n"
        "1. Introduction\n"
        "2. CNN-based detection approach\n"
        "3. Datasets identified in the evidence\n"
        "4. Benefits\n"
        "5. Limitations and challenges\n"
        "6. Applications\n"
        "7. Conclusion\n"
        "8. References\n\n"

        "Use numbered citations only when supported. Include each "
        "unique reference once. If no source supports a statement, "
        "do not attach a citation to it."
    ),
    expected_output=(
        "A structured Markdown report with evidence-supported "
        "statements, valid numbered citations and exact source URLs. "
        "No invented references or unsupported numerical claims."
    ),
    agent=writer,
    context=[research_task],
)


review_task = Task(
    description=(
        "Perform a strict evidence and citation audit of the report.\n\n"

        "INITIAL SEARCH RESULTS:\n{sources}\n\n"

        "Compare the Writer's report with the Research Analyst's "
        "brief, source ledger and initial search results.\n\n"

        "MANDATORY REVIEW CHECKLIST:\n"
        "1. Check every reference title and URL against the supplied "
        "evidence.\n"
        "2. Delete any reference that cannot be found in the evidence.\n"
        "3. Delete citations that point to nonexistent references.\n"
        "4. Remove unsupported statistics, percentages, image counts "
        "and performance claims.\n"
        "5. Remove claims that are not supported by the cited source.\n"
        "6. Do not invent replacement references or URLs.\n"
        "7. Do not claim a webpage was independently verified when "
        "only its search-result description was available.\n"
        "8. Preserve valid exact URLs and remove duplicate references.\n"
        "9. If a section lacks evidence, explain the limitation "
        "instead of filling it with invented facts.\n\n"

        "Return the COMPLETE corrected Markdown report, not just "
        "a review summary. Finish with a References section containing "
        "only verified-against-the-supplied-evidence source entries."
    ),
    expected_output=(
        "A complete corrected Markdown report with an audited "
        "reference list, exact URLs, supported claims and no "
        "fabricated citations."
    ),
    agent=reviewer,
    context=[research_task, write_task],
)

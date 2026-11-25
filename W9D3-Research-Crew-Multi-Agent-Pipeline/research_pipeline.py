from datetime import datetime, timezone


def researcher(topic, sources):
    """Prepare a research brief from supplied source material."""
    findings = [
        {
            "source": source["title"],
            "url": source["url"],
            "evidence": source["evidence"],
        }
        for source in sources
    ]

    return {
        "topic": topic,
        "findings": findings,
        "source_count": len(findings),
        "stage": "Research completed",
    }


def writer(research_brief):
    """Create a Markdown report from the research brief."""
    topic = research_brief["topic"]

    lines = [
        f"# Research Report: {topic}",
        "",
        "## Introduction",
        f"This report summarizes the supplied research material on {topic}.",
        "",
        "## Research Findings",
    ]

    for index, finding in enumerate(research_brief["findings"], start=1):
        lines.extend([
            f"### Finding {index}: {finding['source']}",
            finding["evidence"],
            "",
        ])

    lines.extend([
        "## Conclusion",
        "The report is limited to the evidence supplied to the research stage.",
        "",
        "## References",
    ])

    for index, finding in enumerate(research_brief["findings"], start=1):
        lines.append(f"{index}. [{finding['source']}]({finding['url']})")

    return "\n".join(lines)


def reviewer(report, research_brief):
    """Check report completeness and reference consistency."""
    checks = {
        "title_present": report.startswith("# Research Report:"),
        "findings_present": "## Research Findings" in report,
        "conclusion_present": "## Conclusion" in report,
        "references_present": "## References" in report,
        "all_source_urls_present": all(
            finding["url"] in report
            for finding in research_brief["findings"]
        ),
    }

    return {
        "checks": checks,
        "passed": all(checks.values()),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_pipeline(topic, sources):
    """Run Researcher -> Writer -> Reviewer sequentially."""
    research_brief = researcher(topic, sources)
    report = writer(research_brief)
    review = reviewer(report, research_brief)

    return {
        "research": research_brief,
        "report": report,
        "review": review,
    }

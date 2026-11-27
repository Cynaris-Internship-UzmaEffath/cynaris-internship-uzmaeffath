from pathlib import Path

from crewai import Crew, Process
from ddgs import DDGS

from crew import researcher, writer, reviewer
from tasks import research_task, write_task, review_task


def collect_sources():
    queries = [
        "PlantVillage dataset",
        "crop disease detection CNN",
    ]

    collected = []
    seen_urls = set()

    for query in queries:
        print(f"\nSearching: {query}")

        try:
            results = DDGS().text(
                query,
                backend="bing",
                max_results=5,
            )
        except Exception as error:
            print(f"Search failed for '{query}': {error}")
            continue

        for item in results:
            url = item.get("href", "").strip()

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            collected.append({
                "title": item.get("title", "Untitled"),
                "url": url,
                "description": item.get("body", "No description available."),
            })

    collected = collected[:5]

    if not collected:
        raise RuntimeError(
            "No search results were collected. Check your internet connection."
        )

    source_text = "\n\n".join(
        f"[{index}] {source['title']}\n"
        f"URL: {source['url']}\n"
        f"Description: {source['description']}"
        for index, source in enumerate(collected, start=1)
    )

    print(f"\nCollected {len(collected)} unique sources.")
    print("\n========== SOURCE LIST ==========\n")
    print(source_text)

    return source_text


def main():
    print("\nStarting CrewAI Research Crew...\n")

    sources = collect_sources()

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, write_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff(
        inputs={
            "topic": "AI-based agricultural crop disease detection",
            "sources": sources,
        }
    )

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "crew_generated_report.md"
    output_file.write_text(str(result), encoding="utf-8")

    print("\n========== FINAL RESEARCH REPORT ==========\n")
    print(result)
    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()


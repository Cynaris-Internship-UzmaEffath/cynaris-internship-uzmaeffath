from typing import Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from ddgs import DDGS


class WebSearchInput(BaseModel):
    query: str = Field(
        description="Search query for finding relevant information on the web."
    )


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Search the internet for relevant information. "
        "Returns page titles, URLs, and short descriptions."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        backends = ["bing", "auto"]
        errors = []

        for backend in backends:
            try:
                results = DDGS().text(
                    query,
                    backend=backend,
                    max_results=5
                )

                if results:
                    output = [
                        f"Web search results for: {query}",
                        f"Search backend: {backend}",
                        ""
                    ]

                    for index, result in enumerate(results, start=1):
                        output.append(
                            f"{index}. {result.get('title', 'No title')}\n"
                            f"URL: {result.get('href', 'No URL')}\n"
                            f"Description: {result.get('body', 'No description')}"
                        )

                    return "\n\n".join(output)

            except Exception as error:
                errors.append(f"{backend}: {error}")

        if errors:
            return "Web search unavailable. " + " | ".join(errors)

        return "No search results found from Bing or auto backend."

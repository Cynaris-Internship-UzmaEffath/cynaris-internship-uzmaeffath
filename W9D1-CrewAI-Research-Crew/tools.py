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
        try:
            results = DDGS().text(query, max_results=5)

            if not results:
                return "No search results found."

            output = []

            for index, result in enumerate(results, start=1):
                output.append(
                    f"{index}. {result.get('title', 'No title')}\n"
                    f"URL: {result.get('href', 'No URL')}\n"
                    f"Description: {result.get('body', 'No description')}"
                )

            return "\n\n".join(output)

        except Exception as error:
            return f"Web search failed: {error}"

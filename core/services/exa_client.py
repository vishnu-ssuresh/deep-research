import os
from typing import Any, Optional, Union
from exa_py import Exa
from ..errors import APIKeyError, SearchServiceError

class ExaClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise APIKeyError(
                "Exa API key must be set in EXA_API_KEY environment variable"
            )

        self.client = Exa(api_key=self.api_key)

    def call(
        self,
        query: str,
        num_results: int = 5,
        text: Union[bool, dict[str, int]] = True,
        highlights: Union[bool, dict[str, int]] = False,
    ) -> list[dict[str, Any]]:
        try:
            search_params = {
                "query": query,
                "num_results": num_results,
            }

            if isinstance(text, dict):
                search_params["text"] = text
            elif text:
                search_params["text"] = {"max_characters": 2000}

            if highlights:
                search_params["highlights"] = highlights

            results = self.client.search_and_contents(**search_params)

            formatted_results = []
            for result in results.results:
                formatted_result = {
                    "title": result.title,
                    "url": result.url,
                    "text": getattr(result, "text", None),
                    "highlights": getattr(result, "highlights", None),
                    "published_date": getattr(result, "published_date", None),
                    "author": getattr(result, "author", None),
                }
                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            raise SearchServiceError(f"Exa search failed: {str(e)}") from e

"""Exa client for web search."""

import os
from typing import Any, Optional, Union

from exa_py import Exa

from ..errors import APIKeyError, SearchServiceError


class ExaClient:
    """Client for interacting with Exa search API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Exa client.

        Args:
            api_key: Exa API key. If None, reads from EXA_API_KEY env var
        """
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
        """
        Search the web using Exa.

        Args:
            query: Search query
            num_results: Number of results to return
            text: Whether to include page text. Can be bool or dict with max_characters
            highlights: Whether to include highlights. Can be bool or dict with options

        Returns:
            List of search results with metadata

        Raises:
            SearchServiceError: If the search fails
        """
        try:
            # Build search parameters
            search_params = {
                "query": query,
                "num_results": num_results,
            }

            # Add text options
            if isinstance(text, dict):
                search_params["text"] = text
            elif text:
                search_params["text"] = {"max_characters": 2000}

            # Add highlight options
            if highlights:
                search_params["highlights"] = highlights

            # Execute search
            results = self.client.search_and_contents(**search_params)

            # Format results
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

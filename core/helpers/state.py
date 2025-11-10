"""State schema for the deep research agent."""

from typing import Any, Optional, TypedDict

from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    """State for the research agent graph."""

    # User input and assistant responses
    messages: list[BaseMessage]

    # Research brief outlining what to research
    research_brief: Optional[str]

    # Current queries to execute
    search_queries: list[str]

    # Raw accumulated search results from Exa
    search_results: list[dict[str, Any]]

    # Compressed and distilled summary of all search results
    compressed_findings: Optional[str]

    # Identified gaps in current research
    knowledge_gaps: list[str]

    # Current search iteration (max 5)
    search_iteration: int

    # Whether more context/searches are needed
    needs_more_context: bool


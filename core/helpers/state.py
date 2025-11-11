from typing import Any, Optional, TypedDict

from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    messages: list[BaseMessage]
    research_brief: Optional[str]
    search_queries: list[str]
    search_results: list[dict[str, Any]]
    compressed_findings: Optional[str]
    knowledge_gaps: list[str]
    search_iteration: int
    needs_more_context: bool

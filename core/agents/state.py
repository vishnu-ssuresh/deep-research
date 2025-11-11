from typing import TypedDict

from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    messages: list[BaseMessage]
    research_brief: str | None
    search_queries: list[str]
    search_results: list[dict]
    compressed_findings: str | None
    knowledge_gaps: list[str]
    search_iteration: int
    needs_more_context: bool
    mcp_tool_results: str | None

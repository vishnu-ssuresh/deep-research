"""Pydantic models for structured outputs."""

from pydantic import BaseModel, Field


class ClarifyingQuestions(BaseModel):
    """Model for clarifying questions response."""

    questions: list[str] = Field(
        description="List of 2-4 clarifying questions",
        min_length=2,
        max_length=4,
    )


class SearchQueries(BaseModel):
    """Model for search queries response."""

    queries: list[str] = Field(
        description="List of search queries",
        min_length=1,
    )


class DecisionOutput(BaseModel):
    """Model for decision node output."""

    thought_process: str = Field(
        description="Reasoning about what's been learned and what's still needed"
    )
    compressed_findings: str = Field(
        description="Compressed summary of all search results"
    )
    knowledge_gaps: list[str] = Field(
        description="Identified gaps in research",
        default_factory=list,
    )
    needs_more_context: bool = Field(description="Whether more searches are needed")
    follow_up_queries: list[str] = Field(
        description="Follow-up queries to address gaps",
        default_factory=list,
    )

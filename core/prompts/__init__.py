"""Prompts for the deep research agent."""

from .system_prompts import (
    CLARIFY_SYSTEM_PROMPT,
    COMPRESSION_SYSTEM_PROMPT,
    DECIDE_SYSTEM_PROMPT,
    GENERATE_QUERIES_SYSTEM_PROMPT,
    GENERATE_REPORT_SYSTEM_PROMPT,
    RESEARCH_BRIEF_SYSTEM_PROMPT,
)
from .user_prompts import (
    build_clarify_user_prompt,
    build_compression_user_prompt,
    build_generate_queries_user_prompt,
    build_reflection_user_prompt,
    build_report_user_prompt,
    build_research_brief_user_prompt,
)

__all__ = [
    # System prompts
    "CLARIFY_SYSTEM_PROMPT",
    "RESEARCH_BRIEF_SYSTEM_PROMPT",
    "GENERATE_QUERIES_SYSTEM_PROMPT",
    "COMPRESSION_SYSTEM_PROMPT",
    "DECIDE_SYSTEM_PROMPT",
    "GENERATE_REPORT_SYSTEM_PROMPT",
    # User prompt builders
    "build_clarify_user_prompt",
    "build_research_brief_user_prompt",
    "build_generate_queries_user_prompt",
    "build_compression_user_prompt",
    "build_reflection_user_prompt",
    "build_report_user_prompt",
]

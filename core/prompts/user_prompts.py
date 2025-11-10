"""User prompt generation functions for the deep research agent."""

from typing import Any


def build_clarify_user_prompt(original_query: str) -> str:
    """Build user prompt for clarifying questions generation."""
    return (
        f"User's research query: {original_query}\n\nGenerate 2-4 clarifying questions."
    )


def build_research_brief_user_prompt(messages: list[Any]) -> str:
    """Build user prompt for research brief generation."""
    # Build context from all messages
    context_parts = []
    for msg in messages:
        if msg.type == "human":
            context_parts.append(f"USER: {msg.content}")
        elif msg.type == "ai":
            context_parts.append(f"ASSISTANT: {msg.content}")

    context = "\n\n".join(context_parts)

    return f"""Based on the following conversation, create a comprehensive research brief.

{context}

Create a research brief that clearly outlines:
1. Research objective - what we're trying to find out
2. Key topics and subtopics to investigate
3. Expected scope and depth
4. Any specific angles or perspectives to focus on"""


def build_generate_queries_user_prompt(
    research_brief: str,
    search_iteration: int,
    num_queries: int = 5,
    compressed_findings: str = "",
    knowledge_gaps: list[str] | None = None,
) -> str:
    """Build user prompt for search query generation."""
    if search_iteration == 0:
        # Initial queries
        return f"""Research Brief:
{research_brief}

Generate {num_queries} diverse, targeted search queries to gather comprehensive information for this research.

Remember:
- Each query should focus on ONE specific aspect
- Avoid generating similar queries
- Make queries self-contained with necessary context"""
    else:
        # Follow-up queries
        gaps_text = "\n".join(f"- {gap}" for gap in (knowledge_gaps or []))

        return f"""Research Brief:
{research_brief}

Current Findings Summary:
{compressed_findings}

Knowledge Gaps:
{gaps_text}

Generate up to {num_queries} follow-up search queries ONLY if needed to address critical knowledge gaps.

Remember:
- Each query should address ONE specific gap
- Don't generate similar queries
- Make queries self-contained with necessary context
- If the current findings are sufficient, generate fewer queries or none"""


def build_reflection_user_prompt(
    research_brief: str,
    search_results: list[dict[str, Any]],
    search_iteration: int,
) -> str:
    """Build user prompt for reflection/decision analysis."""
    # Format search results for analysis
    results_text = []
    for i, result in enumerate(search_results, 1):
        result_info = f"""
Result {i}:
- Query: {result.get("query", "N/A")}
- Title: {result.get("title", "N/A")}
- URL: {result.get("url", "N/A")}
- Content: {result.get("text", "N/A")[:500]}...
"""
        results_text.append(result_info)

    results_summary = "\n".join(results_text[:30])  # Limit to avoid token overflow

    return f"""Research Brief:
{research_brief}

Search Results from {search_iteration} iteration(s):
{results_summary}

Total results collected: {len(search_results)}

Analyze these results carefully:

1. Think through what you've learned and whether it's sufficient to answer the user's question
2. Create a compressed summary of all findings
3. Identify any CRITICAL knowledge gaps (not minor details, but essential information needed)
4. Decide if the current information is sufficient to generate a comprehensive report
5. ONLY generate follow-up queries if there are critical gaps that prevent answering the question

Remember:
- If the summaries are sufficient to answer the user's question, set needs_more_context to false and leave follow_up_queries empty
- Each follow-up query should focus on ONE specific knowledge gap
- Don't generate similar queries
- Make queries self-contained with necessary context for web search
- We're on iteration {search_iteration} of max 5, so be judicious about continuing"""


def build_report_user_prompt(
    original_query: str,
    research_brief: str,
    compressed_findings: str,
    search_results: list[dict[str, Any]],
) -> str:
    """Build user prompt for final report generation."""
    # Format search results as sources for citation
    sources_text = []
    for i, result in enumerate(search_results, 1):
        source_entry = f"""[{i}] {result.get("title", "Untitled")}
    URL: {result.get("url", "N/A")}
    Query: {result.get("query", "N/A")}
    Content Preview: {result.get("text", "N/A")[:300]}...
"""
        sources_text.append(source_entry)

    sources_summary = "\n".join(sources_text[:50])  # Limit for token management

    return f"""Original Research Query: {original_query}

Research Brief:
{research_brief}

Compressed Findings from {len(search_results)} sources:
{compressed_findings}

Available Sources for Citation:
{sources_summary}

Create a comprehensive deep research report with the following structure:

# [Research Topic Title]

## Executive Summary
Brief overview of key findings (2-3 paragraphs)

## Introduction
- Background and context
- Research objectives
- Scope of investigation

## Key Findings
Organized by themes/topics with clear subheadings
- Include specific data, facts, and insights
- Cite sources using markdown format: [Title](URL)

## Detailed Analysis
Deep dive into the findings with:
- Multiple sections based on research brief topics
- Evidence-based insights
- Connections between different findings
- Citations throughout

## Implications
What do these findings mean?
- Practical implications
- Future considerations

## Conclusion
Summary and final thoughts

## Sources
List all cited sources with full URLs

Important:
- Use markdown formatting (headers, lists, bold, etc.)
- Cite sources inline as [Source Title](URL)
- Be thorough and professional
- Address all aspects of the research brief
- Make it publication-ready"""

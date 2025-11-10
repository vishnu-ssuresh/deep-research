"""User prompt generation functions for the deep research agent."""

from typing import Any, Optional


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
    knowledge_gaps: Optional[list[str]] = None,
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
        title = result.get("title", "Untitled")
        url = result.get("url", "N/A")
        content = result.get("text", "N/A")[:500]

        source_entry = f"""Source {i}:
Title: {title}
URL: {url}
Content: {content}...
"""
        sources_text.append(source_entry)

    sources_summary = "\n".join(sources_text[:50])  # Limit for token management

    return f"""Generate a high-quality answer to the user's question based on the provided summaries.

User's Research Question:
{original_query}

Research Context:
{research_brief}

Summaries from {len(search_results)} sources:
{compressed_findings}

Available Sources (MUST be cited in markdown format):
{sources_summary}

CRITICAL INSTRUCTIONS:
- Answer the user's question comprehensively and in-depth using the summaries and sources
- Use markdown formatting for structure (headings, lists, bold, etc.)
- Cite sources naturally throughout using markdown: [Source Title](URL)
- Every claim should be backed by a source citation
- Organize information logically - use whatever structure makes sense for this topic
- Be thorough and detailed - each major section should be 3-5 paragraphs with specific examples
- Include concrete details: statistics, dates, specific achievements, quotes, and examples
- Provide deep insights and analysis, not just surface-level facts
- Elaborate on key points with context and explanation
- Connect ideas and show relationships between different findings

DEPTH REQUIREMENTS:
- Don't just list facts - explain their significance and context
- Include specific numbers, dates, and concrete examples when possible
- Expand on important points with multiple supporting details
- Provide nuanced analysis that demonstrates deep understanding

DO NOT mention that you are the final step of a research process. Write naturally as if you're an expert providing a comprehensive, detailed explanation."""

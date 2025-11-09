"""LangGraph workflow definition for the deep research agent."""

from langgraph.graph import END, StateGraph

from .nodes import (
    clarify_node,
    decide_node,
    generate_queries_node,
    generate_report_node,
    research_brief_node,
    search_node,
)
from .state import ResearchState


def should_continue_searching(state: ResearchState) -> str:
    """Determine whether to continue searching or generate report."""
    if state["needs_more_context"] and state["search_iteration"] < 5:
        return "search"
    return "generate_report"


def create_graph():
    """Create and compile the research agent graph."""

    # Initialize the graph
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("research_brief", research_brief_node)
    workflow.add_node("generate_queries", generate_queries_node)
    workflow.add_node("search", search_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("generate_report", generate_report_node)

    # Add edges
    workflow.set_entry_point("clarify")
    workflow.add_edge("clarify", "research_brief")
    workflow.add_edge("research_brief", "generate_queries")
    workflow.add_edge("generate_queries", "search")
    workflow.add_edge("search", "decide")

    # Conditional edge: continue searching or generate report
    workflow.add_conditional_edges(
        "decide",
        should_continue_searching,
        {
            "search": "search",
            "generate_report": "generate_report",
        }
    )

    workflow.add_edge("generate_report", END)

    # Compile and return
    return workflow.compile()


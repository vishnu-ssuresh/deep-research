"""LangGraph workflow definition for the deep research agent."""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    clarify_node,
    generate_queries_node,
    generate_report_node,
    reflection_node,
    research_brief_node,
    search_node,
)
from .state import ResearchState


def should_continue_searching(state: ResearchState) -> str:
    """
    Determine whether to continue searching or generate report.
    
    Returns:
        - "generate_queries": If more context needed and under iteration limit
        - "generate_report": If research is complete or iteration limit reached
    """
    if state.get("needs_more_context", False) and state.get("search_iteration", 0) < 5:
        return "generate_queries"
    return "generate_report"


def create_graph():
    """
    Create and compile the research agent graph.
    
    Workflow:
        1. clarify: Ask clarifying questions to understand research needs
        2. research_brief: Generate comprehensive research brief
        3. generate_queries: Create targeted search queries
        4. search: Execute searches using Exa API
        5. reflect: Analyze results, identify gaps, decide next steps
        6. [Conditional] Back to generate_queries OR proceed to generate_report
        7. generate_report: Create final comprehensive markdown report
    
    The graph supports iterative research with up to 5 search iterations,
    refining queries based on identified knowledge gaps.
    """
    # Initialize the graph
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("research_brief", research_brief_node)
    workflow.add_node("generate_queries", generate_queries_node)
    workflow.add_node("search", search_node)
    workflow.add_node("reflect", reflection_node)
    workflow.add_node("generate_report", generate_report_node)

    # Set entry point
    workflow.add_edge(START, "clarify")
    
    # Add linear edges for main flow
    workflow.add_edge("clarify", "research_brief")
    workflow.add_edge("research_brief", "generate_queries")
    workflow.add_edge("generate_queries", "search")
    workflow.add_edge("search", "reflect")

    # Add conditional edge from reflect
    # - If more context needed: loop back to generate_queries
    # - Otherwise: proceed to generate_report
    workflow.add_conditional_edges(
        "reflect",
        should_continue_searching,
        {
            "generate_queries": "generate_queries",
            "generate_report": "generate_report",
        }
    )

    # Add final edge to END
    workflow.add_edge("generate_report", END)

    # Compile and return the graph
    return workflow.compile()


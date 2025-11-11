from langgraph.graph import END, START, StateGraph

from .nodes import (
    clarify_node,
    compression_node,
    generate_queries_node,
    generate_report_node,
    mcp_tool_node,
    reflection_node,
    research_brief_node,
    save_pdf_node,
    search_node,
)
from .state import ResearchState


def should_continue_searching(state: ResearchState) -> str:
    search_iteration = state.get("search_iteration", 0)
    needs_more_context = state.get("needs_more_context", False)

    # Minimum 3 iterations
    if search_iteration < 3:
        return "generate_queries"

    if needs_more_context and search_iteration < 5:
        return "generate_queries"

    return "generate_report"


def create_graph() -> StateGraph:
    workflow = StateGraph(ResearchState)

    workflow.add_node("clarify", clarify_node)
    workflow.add_node("research_brief", research_brief_node)
    workflow.add_node("generate_queries", generate_queries_node)
    workflow.add_node("search", search_node)
    workflow.add_node("mcp_tools", mcp_tool_node)
    workflow.add_node("compress", compression_node)
    workflow.add_node("reflect", reflection_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("save_pdf", save_pdf_node)

    workflow.add_edge(START, "clarify")
    workflow.add_edge("clarify", "research_brief")
    workflow.add_edge("research_brief", "generate_queries")
    workflow.add_edge("generate_queries", "search")
    workflow.add_edge("generate_queries", "mcp_tools")
    workflow.add_edge("search", "compress")
    workflow.add_edge("mcp_tools", "compress")
    workflow.add_edge("compress", "reflect")

    workflow.add_conditional_edges(
        "reflect",
        should_continue_searching,
        {
            "generate_queries": "generate_queries",
            "generate_report": "generate_report",
        },
    )

    workflow.add_edge("generate_report", "save_pdf")
    workflow.add_edge("save_pdf", END)

    return workflow.compile()

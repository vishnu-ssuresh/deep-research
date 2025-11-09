"""Main CLI interface for the deep research agent."""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from core.graph import create_graph


def main():
    """Run the deep research agent."""
    # Load environment variables
    load_dotenv()

    # Verify API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    if not os.getenv("EXA_API_KEY"):
        raise ValueError("EXA_API_KEY not found in environment variables")

    # Get user query
    print("=== Deep Research Agent ===\n")
    query = input("Enter your research query: ").strip()

    if not query:
        print("No query provided. Exiting.")
        return

    # Create the graph
    graph = create_graph()

    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "research_brief": None,
        "search_queries": [],
        "search_results": [],
        "compressed_findings": None,
        "knowledge_gaps": [],
        "search_iteration": 0,
        "needs_more_context": True,
    }

    # Run the graph
    print("\nStarting research...\n")
    final_state = graph.invoke(initial_state)

    # Display the final report
    print("\n=== Research Report ===\n")
    for message in final_state["messages"]:
        if message.type == "assistant":
            print(message.content)

    print("\n=== Research Complete ===")


if __name__ == "__main__":
    main()


import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from core.exceptions import APIKeyException
from core.helpers import create_graph


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise APIKeyException("OPENAI_API_KEY not found in environment variables")
    if not os.getenv("EXA_API_KEY"):
        raise APIKeyException("EXA_API_KEY not found in environment variables")

    print("=== Deep Research Agent ===\n")
    query = input("Enter your research query: ").strip()

    if not query:
        print("No query provided. Exiting.")
        return

    graph = create_graph()

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

    print("\nStarting research...\n")
    final_state = graph.invoke(initial_state)

    print("\n=== Research Report ===\n")
    for message in final_state["messages"]:
        if message.type == "assistant":
            print(message.content)

    print("\n=== Research Complete ===")


if __name__ == "__main__":
    main()

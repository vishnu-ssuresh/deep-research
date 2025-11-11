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

    topic = input("Enter your research topic: ")

    graph = create_graph()

    initial_state = {
        "messages": [HumanMessage(content=topic)],
        "research_brief": None,
        "search_queries": [],
        "search_results": [],
        "compressed_findings": None,
        "knowledge_gaps": [],
        "search_iteration": 0,
        "needs_more_context": True,
    }

    print("Starting deep research...")

    final_state = graph.invoke(initial_state)

    for message in final_state["messages"]:
        if message.type == "assistant":
            print(message.content)

    print("Deep research complete")


if __name__ == "__main__":
    main()

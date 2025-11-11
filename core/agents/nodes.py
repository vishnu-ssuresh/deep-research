from langchain_core.messages import AIMessage, HumanMessage

from ..exceptions import NodeException
from ..models import ClarifyingQuestions, DecisionOutput, SearchQueries
from ..prompts import (
    CLARIFY_SYSTEM_PROMPT,
    COMPRESSION_SYSTEM_PROMPT,
    DECIDE_SYSTEM_PROMPT,
    FILENAME_GENERATION_SYSTEM_PROMPT,
    GENERATE_QUERIES_SYSTEM_PROMPT,
    GENERATE_REPORT_SYSTEM_PROMPT,
    RESEARCH_BRIEF_SYSTEM_PROMPT,
    build_clarify_user_prompt,
    build_compression_user_prompt,
    build_filename_user_prompt,
    build_generate_queries_user_prompt,
    build_reflection_user_prompt,
    build_report_user_prompt,
    build_research_brief_user_prompt,
)
from ..services import ExaClient, OpenAIClient
from ..utils import save_report_to_disk
from .state import ResearchState


def clarify_node(state: ResearchState) -> ResearchState:
    messages = state["messages"]
    original_query = messages[0].content if messages else ""

    llm = OpenAIClient()

    user_prompt = build_clarify_user_prompt(original_query)

    response = llm.call(
        system_prompt=CLARIFY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.5,
        response_format=ClarifyingQuestions,
    )

    questions = response.questions

    questions_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))

    messages.append(
        AIMessage(
            content=f"To better understand your research needs, I have a few questions:\n\n{questions_text}"
        )
    )

    print(f"\n{messages[-1].content}\n")

    answers = []
    for i, question in enumerate(questions):
        answer = input(f"Answer {i + 1}: ").strip()
        answers.append(f"Q: {question}\nA: {answer}")

    answers_text = "\n\n".join(answers)
    messages.append(HumanMessage(content=f"Here are my answers:\n\n{answers_text}"))

    return {"messages": messages}


def research_brief_node(state: ResearchState) -> ResearchState:
    messages = state["messages"]

    llm = OpenAIClient()

    user_prompt = build_research_brief_user_prompt(messages)

    research_brief = llm.call(
        system_prompt=RESEARCH_BRIEF_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.5,
    )

    print(f"\n=== Research Brief ===\n{research_brief}\n")

    return {
        "research_brief": research_brief,
    }


def generate_queries_node(state: ResearchState) -> ResearchState:
    research_brief = state.get("research_brief", "")
    search_iteration = state.get("search_iteration", 0)

    llm = OpenAIClient()

    if search_iteration == 0:
        iteration_context = ""
        num_queries = 5
    else:
        iteration_context = f" (iteration {search_iteration + 1})"
        num_queries = 3

    system_prompt = GENERATE_QUERIES_SYSTEM_PROMPT.format(
        iteration_context=iteration_context,
        num_queries=num_queries,
    )

    user_prompt = build_generate_queries_user_prompt(
        research_brief=research_brief,
        search_iteration=search_iteration,
        num_queries=num_queries,
        compressed_findings=state.get("compressed_findings", ""),
        knowledge_gaps=state.get("knowledge_gaps", []),
    )

    response = llm.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        response_format=SearchQueries,
    )

    queries = response.queries

    # TODO: comment out
    print(f"\n=== Generated Search Queries (Iteration {search_iteration + 1}) ===")
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
    print()

    return {
        "search_queries": queries,
    }


def search_node(state: ResearchState) -> ResearchState:
    search_queries = state.get("search_queries", [])
    search_results = state.get("search_results", [])
    search_iteration = state.get("search_iteration", 0)

    if not search_queries:
        raise NodeException("No search queries found in state")

    exa = ExaClient()

    print(f"\n=== Executing Searches (Iteration {search_iteration + 1}) ===")

    for i, query in enumerate(search_queries, 1):
        print(f"[{i}/{len(search_queries)}] Searching: {query}")

        try:
            results = exa.call(
                query=query,
                num_results=5,
                text={"max_characters": 2000},
            )

            # TODO: comment out
            for result in results:
                result["query"] = query
                result["iteration"] = search_iteration + 1

            search_results.extend(results)

        except Exception:
            # TODO: handle error
            continue

    print(f"\nTotal results accumulated: {len(search_results)}\n")

    return {
        "search_queries": search_queries,
        "search_results": search_results,
        "search_iteration": search_iteration + 1,
    }


def compression_node(state: ResearchState) -> ResearchState:
    research_brief = state.get("research_brief", "")
    search_results = state.get("search_results", [])
    search_iteration = state.get("search_iteration", 0)

    llm = OpenAIClient()

    user_prompt = build_compression_user_prompt(
        research_brief=research_brief,
        search_results=search_results,
        search_iteration=search_iteration,
    )

    compressed_findings = llm.call(
        system_prompt=COMPRESSION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    )

    return {
        "compressed_findings": compressed_findings,
    }


def reflection_node(state: ResearchState) -> ResearchState:
    research_brief = state.get("research_brief", "")
    compressed_findings = state.get("compressed_findings", "")
    search_iteration = state.get("search_iteration", 0)

    llm = OpenAIClient()

    system_prompt = DECIDE_SYSTEM_PROMPT.format(num_iterations=search_iteration)

    user_prompt = build_reflection_user_prompt(
        research_brief=research_brief,
        compressed_findings=compressed_findings,
        search_iteration=search_iteration,
    )

    response = llm.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
        response_format=DecisionOutput,
    )

    print("💭 Thought Process:")
    print(f"{response.thought_process}\n")

    return {
        "search_queries": response.follow_up_queries
        if response.needs_more_context
        else [],
        "knowledge_gaps": response.knowledge_gaps,
        "needs_more_context": response.needs_more_context and search_iteration < 5,
    }


def generate_report_node(state: ResearchState) -> ResearchState:
    research_brief = state.get("research_brief", "")
    compressed_findings = state.get("compressed_findings", "")
    search_results = state.get("search_results", [])
    messages = state["messages"]

    original_query = messages[0].content if messages else "Research Report"

    llm = OpenAIClient()

    user_prompt = build_report_user_prompt(
        original_query=original_query,
        research_brief=research_brief,
        compressed_findings=compressed_findings,
        search_results=search_results,
    )

    report = llm.call(
        system_prompt=GENERATE_REPORT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.4,
        model="gpt-4o",
    )

    messages.append(AIMessage(content=report))

    return {
        "messages": messages,
    }


def save_pdf_node(state: ResearchState) -> ResearchState:
    messages = state["messages"]

    report_content = None
    for msg in reversed(messages):
        if msg.type == "ai" and len(msg.content) > 1000:
            report_content = msg.content
            break

    if not report_content:
        raise NodeException("No report content found in messages (looking for AI message >1000 chars)")

    original_query = messages[0].content if messages else "research_report"

    llm = OpenAIClient()
    user_prompt = build_filename_user_prompt(original_query)

    filename = llm.call(
        system_prompt=FILENAME_GENERATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
    ).strip()

    save_report_to_disk(
        report_content=report_content,
        filename=filename,
        reports_dir="reports",
    )

    return state


def mcp_tool_node(state: ResearchState) -> ResearchState:
    """
    Connects with Model Context Protocol (MCP) servers to augment research.
    """

    # Basic MCP integration (https://docs.langchain.com/oss/python/langchain/mcp)
    # server_configs = {}
    # mcp_client = MultiServerMCPClient(server_configs)
    # tools = await mcp_client.get_tools()
    # agent = create_agent("gpt-4o-mini", tools=tools)

    # Return dummy results
    return {
        "mcp_tool_results": "MCP tool results",
    }


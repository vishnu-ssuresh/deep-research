import os

import markdown
from langchain_core.messages import AIMessage, HumanMessage
from xhtml2pdf import pisa

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
from .state import ResearchState


def clarify_node(state: ResearchState) -> ResearchState:
    """
    Ask clarifying questions about the user's query.
    """

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
    """
    Generate a research brief based on query and clarifications.
    """

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
    """
    Generate search queries based on research brief.
    """

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
    """
    Execute Exa searches.
    """

    search_queries = state.get("search_queries", [])
    search_results = state.get("search_results", [])
    search_iteration = state.get("search_iteration", 0)

    if not search_queries:
        # TODO: handle error
        raise ValueError("No search queries found in state")

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
    """
    Compress all search results into a clean, comprehensive summary.
    """
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
    """
    Reflect on compressed findings, identify knowledge gaps, and decide next steps.
    """

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
    """
    Generate final comprehensive markdown report with citations.
    """

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
    """
    Convert markdown report to PDF and save locally.
    """

    messages = state["messages"]

    report_content = None
    for msg in reversed(messages):
        if msg.type == "ai" and len(msg.content) > 1000:
            report_content = msg.content
            break

    if not report_content:
        raise ValueError("No report content found in messages (looking for AI message >1000 chars)")

    original_query = messages[0].content if messages else "research_report"

    llm = OpenAIClient()

    user_prompt = build_filename_user_prompt(original_query)

    filename = llm.call(
        system_prompt=FILENAME_GENERATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
    ).strip()

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    markdown_path = os.path.join(reports_dir, f"{filename}.md")
    pdf_path = os.path.join(reports_dir, f"{filename}.pdf")

    try:
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(report_content)
    except Exception as e:
        # TODO: handle error
        raise ValueError(f"Failed to save markdown: {str(e)}")

    try:
        html_content = markdown.markdown(report_content, extensions=["extra", "codehilite", "tables", "toc"])
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: A4;
                    margin: 2cm;
                }}
                body {{
                    font-family: 'Helvetica', 'Arial', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 100%;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    margin-top: 30px;
                }}
                h2 {{
                    color: #34495e;
                    border-bottom: 2px solid #95a5a6;
                    padding-bottom: 8px;
                    margin-top: 25px;
                }}
                h3 {{
                    color: #34495e;
                    margin-top: 20px;
                }}
                a {{
                    color: #3498db;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                p {{
                    margin: 12px 0;
                    text-align: justify;
                }}
                ul, ol {{
                    margin: 12px 0;
                    padding-left: 30px;
                }}
                li {{
                    margin: 8px 0;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                blockquote {{
                    border-left: 4px solid #3498db;
                    padding-left: 20px;
                    margin: 20px 0;
                    color: #555;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        with open(pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(styled_html, dest=pdf_file)

        if pisa_status.err:
            # TODO: handle error
            raise ValueError("Failed to generate PDF")

    except Exception as e:
        # TODO: handle error
        raise ValueError(f"Failed to generate PDF: {str(e)}")

    return state

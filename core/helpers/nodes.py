"""Node implementations for the deep research agent."""

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

    This node:
    1. Extracts the user's original query
    2. Uses LLM to generate 2-4 clarifying questions
    3. Interacts with user to get answers
    4. Adds Q&A to messages
    """
    # Extract the original query
    messages = state["messages"]
    original_query = messages[0].content if messages else ""

    # Initialize OpenAI client
    llm = OpenAIClient()

    # Generate clarifying questions
    user_prompt = build_clarify_user_prompt(original_query)

    response = llm.call(
        system_prompt=CLARIFY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.7,
        response_format=ClarifyingQuestions,
    )

    questions = response.questions

    # Format questions for display
    questions_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))

    # Add questions as AI message
    messages.append(
        AIMessage(
            content=f"To better understand your research needs, I have a few questions:\n\n{questions_text}"
        )
    )

    # Get user answers (interactive)
    print(f"\n{messages[-1].content}\n")

    answers = []
    for i, question in enumerate(questions):
        answer = input(f"Answer {i + 1}: ").strip()
        answers.append(f"Q: {question}\nA: {answer}")

    # Add answers as human message
    answers_text = "\n\n".join(answers)
    messages.append(HumanMessage(content=f"Here are my answers:\n\n{answers_text}"))

    return {"messages": messages}


def research_brief_node(state: ResearchState) -> ResearchState:
    """
    Generate a research brief based on query and clarifications.

    This node:
    1. Extracts the original query and clarifying Q&A
    2. Uses LLM to generate a comprehensive research brief
    3. Stores the brief in state
    """
    messages = state["messages"]

    # Initialize OpenAI client
    llm = OpenAIClient()

    # Generate research brief
    user_prompt = build_research_brief_user_prompt(messages)

    research_brief = llm.call(
        system_prompt=RESEARCH_BRIEF_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.5,
    )

    print(f"\n=== Research Brief ===\n{research_brief}\n")

    return {
        "messages": messages,
        "research_brief": research_brief,
    }


def generate_queries_node(state: ResearchState) -> ResearchState:
    """
    Generate search queries based on research brief.

    This node:
    1. Reads the research brief
    2. Generates 3-5 targeted search queries
    3. On iterations > 0, uses follow_up_queries from decide_node
    4. Stores queries in state
    """
    research_brief = state.get("research_brief", "")
    search_iteration = state.get("search_iteration", 0)

    # Initialize OpenAI client
    llm = OpenAIClient()

    # Determine if this is initial or follow-up queries
    if search_iteration == 0:
        iteration_context = ""
        num_queries = 5
    else:
        iteration_context = f" (iteration {search_iteration + 1})"
        num_queries = 3

    # Build user prompt
    user_prompt = build_generate_queries_user_prompt(
        research_brief=research_brief,
        search_iteration=search_iteration,
        num_queries=num_queries,
        compressed_findings=state.get("compressed_findings", ""),
        knowledge_gaps=state.get("knowledge_gaps", []),
    )

    # Format the system prompt with context
    system_prompt = GENERATE_QUERIES_SYSTEM_PROMPT.format(
        iteration_context=iteration_context,
        num_queries=num_queries,
    )

    response = llm.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        response_format=SearchQueries,
    )

    queries = response.queries

    print(f"\n=== Generated Search Queries (Iteration {search_iteration + 1}) ===")
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
    print()

    return {
        "messages": state["messages"],
        "research_brief": research_brief,
        "search_queries": queries,
    }


def search_node(state: ResearchState) -> ResearchState:
    """
    Execute Exa searches and accumulate results.

    This node:
    1. Takes search queries from state
    2. Executes searches using ExaClient
    3. Accumulates results in state
    4. Increments search iteration counter
    """
    search_queries = state.get("search_queries", [])
    search_results = state.get("search_results", [])
    search_iteration = state.get("search_iteration", 0)

    if not search_queries:
        print("Warning: No search queries found in state")
        return state

    # Initialize Exa client
    exa = ExaClient()

    print(f"\n=== Executing Searches (Iteration {search_iteration + 1}) ===")

    # Execute each search query
    for i, query in enumerate(search_queries, 1):
        print(f"[{i}/{len(search_queries)}] Searching: {query}")

        try:
            results = exa.call(
                query=query,
                num_results=5,
                text={"max_characters": 2000},
            )

            # Add query context to each result
            for result in results:
                result["query"] = query
                result["iteration"] = search_iteration + 1

            search_results.extend(results)
            print(f"    ✓ Found {len(results)} results")

        except Exception as e:
            print(f"    ✗ Search failed: {str(e)}")
            continue

    print(f"\nTotal results accumulated: {len(search_results)}\n")

    return {
        "messages": state["messages"],
        "research_brief": state.get("research_brief"),
        "search_queries": search_queries,
        "search_results": search_results,
        "search_iteration": search_iteration + 1,
    }


def compression_node(state: ResearchState) -> ResearchState:
    """
    Compress all search results into a clean, comprehensive summary.

    This node:
    1. Takes all accumulated search results
    2. Distills them into a clean, comprehensive summary
    3. Preserves all important information and findings
    """
    research_brief = state.get("research_brief", "")
    search_results = state.get("search_results", [])
    search_iteration = state.get("search_iteration", 0)

    # Initialize OpenAI client
    llm = OpenAIClient()

    # Build user prompt
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

    print(f"\n{'=' * 80}")
    print(f"📝 COMPRESSION (After Iteration {search_iteration})")
    print(f"{'=' * 80}\n")
    print(f"Compressed {len(search_results)} search results into summary\n")

    return {
        "messages": state["messages"],
        "research_brief": research_brief,
        "search_queries": state.get("search_queries", []),
        "search_results": search_results,
        "compressed_findings": compressed_findings,
        "knowledge_gaps": state.get("knowledge_gaps", []),
        "search_iteration": search_iteration,
        "needs_more_context": state.get("needs_more_context", True),
    }


def reflection_node(state: ResearchState) -> ResearchState:
    """
    Reflect on compressed findings, identify knowledge gaps, and decide next steps.

    This node:
    1. Analyzes the compressed findings against research brief
    2. Identifies knowledge gaps based on research brief
    3. Decides if more context is needed (max 5 iterations)
    4. If yes, generates follow-up queries to address gaps
    """
    research_brief = state.get("research_brief", "")
    compressed_findings = state.get("compressed_findings", "")
    search_iteration = state.get("search_iteration", 0)

    # Initialize OpenAI client
    llm = OpenAIClient()

    # Build user prompt using compressed findings
    user_prompt = build_reflection_user_prompt(
        research_brief=research_brief,
        compressed_findings=compressed_findings,
        search_iteration=search_iteration,
    )

    # Format system prompt with iteration info
    system_prompt = DECIDE_SYSTEM_PROMPT.format(num_iterations=search_iteration)

    response = llm.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
        response_format=DecisionOutput,
    )

    # Print reflection analysis
    print(f"\n{'=' * 80}")
    print(f"🤔 AGENT THINKING (After Iteration {search_iteration})")
    print(f"{'=' * 80}\n")

    print("💭 Thought Process:")
    print(f"{response.thought_process}\n")

    if response.knowledge_gaps:
        print("🔍 Knowledge Gaps Identified:")
        for i, gap in enumerate(response.knowledge_gaps, 1):
            print(f"  {i}. {gap}")
        print()

    decision = (
        "✅ Sufficient - Moving to report generation"
        if not response.needs_more_context
        else "🔄 Need more information"
    )
    print(f"🎯 Decision: {decision}\n")

    if response.needs_more_context and response.follow_up_queries:
        print("🔎 Follow-up Search Queries:")
        for i, query in enumerate(response.follow_up_queries, 1):
            print(f"  {i}. {query}")
        print()

    print(f"{'=' * 80}\n")

    return {
        "messages": state["messages"],
        "research_brief": research_brief,
        "search_queries": response.follow_up_queries
        if response.needs_more_context
        else [],
        "search_results": state["search_results"],
        "compressed_findings": compressed_findings,  # Keep from compression_node
        "knowledge_gaps": response.knowledge_gaps,
        "search_iteration": search_iteration,
        "needs_more_context": response.needs_more_context and search_iteration < 5,
    }


def generate_report_node(state: ResearchState) -> ResearchState:
    """
    Generate final comprehensive markdown report with citations.

    This node:
    1. Takes research brief and compressed findings
    2. Formats all search results for reference
    3. Generates structured markdown report with citations
    4. Adds report as assistant message
    """
    research_brief = state.get("research_brief", "")
    compressed_findings = state.get("compressed_findings", "")
    search_results = state.get("search_results", [])
    messages = state["messages"]

    # Get original query for report title
    original_query = messages[0].content if messages else "Research Report"

    # Initialize OpenAI client
    llm = OpenAIClient()

    # Build user prompt
    user_prompt = build_report_user_prompt(
        original_query=original_query,
        research_brief=research_brief,
        compressed_findings=compressed_findings,
        search_results=search_results,
    )

    print("\n=== Generating Final Report ===")
    print("This may take a moment...\n")

    report = llm.call(
        system_prompt=GENERATE_REPORT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.6,
        model="gpt-4o",  # Use more capable model for report generation
    )

    # Add report as assistant message
    messages.append(AIMessage(content=report))

    print("✓ Report generation complete!\n")

    return {
        "messages": messages,
        "research_brief": research_brief,
        "compressed_findings": compressed_findings,
        "search_results": search_results,
        "search_queries": state.get("search_queries", []),
        "knowledge_gaps": state.get("knowledge_gaps", []),
        "search_iteration": state.get("search_iteration", 0),
        "needs_more_context": False,  # Done with research
    }


def save_pdf_node(state: ResearchState) -> ResearchState:
    """
    Convert markdown report to PDF and save locally.

    This node:
    1. Extracts the markdown report from messages
    2. Uses LLM to generate a clean filename
    3. Converts markdown to HTML
    4. Converts HTML to PDF
    5. Saves both markdown and PDF versions
    """
    messages = state["messages"]

    # Find the report (last AI message with substantial content)
    report_content = None
    for msg in reversed(messages):
        if msg.type == "ai" and len(msg.content) > 1000:
            report_content = msg.content
            break

    if not report_content:
        print("⚠️ Warning: No report found to convert to PDF")
        return state

    # Get original query
    original_query = messages[0].content if messages else "research_report"

    # Initialize OpenAI client
    llm = OpenAIClient()

    # Use LLM to generate a clean filename
    user_prompt = build_filename_user_prompt(original_query)

    filename = llm.call(
        system_prompt=FILENAME_GENERATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.3,
    ).strip()

    # Create reports directory if it doesn't exist
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    markdown_path = os.path.join(reports_dir, f"{filename}.md")
    pdf_path = os.path.join(reports_dir, f"{filename}.pdf")

    print("\n=== Saving Report ===")

    # Save markdown version
    try:
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"✓ Markdown saved: {markdown_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to save markdown: {str(e)}")
        return state

    # Convert markdown to HTML
    try:
        html_content = markdown.markdown(
            report_content, extensions=["extra", "codehilite", "tables", "toc"]
        )

        # Add CSS styling for better PDF output
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

        # Convert HTML to PDF using xhtml2pdf
        with open(pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(styled_html, dest=pdf_file)

        if pisa_status.err:
            print("⚠️ Warning: PDF generation had errors")
            print(f"   Markdown report is still available at: {markdown_path}")
        else:
            print(f"✓ PDF saved: {pdf_path}")

    except Exception as e:
        print(f"⚠️ Warning: Failed to generate PDF: {str(e)}")
        print(f"   Markdown report is still available at: {markdown_path}")

    print()

    return state

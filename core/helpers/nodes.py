"""Node implementations for the deep research agent."""

from langchain_core.messages import AIMessage, HumanMessage

from ..models import ClarifyingQuestions, DecisionOutput, SearchQueries
from ..prompts import (
    CLARIFY_SYSTEM_PROMPT,
    DECIDE_SYSTEM_PROMPT,
    GENERATE_QUERIES_SYSTEM_PROMPT,
    RESEARCH_BRIEF_SYSTEM_PROMPT,
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
    user_prompt = f"User's research query: {original_query}\n\nGenerate 2-4 clarifying questions."
    
    response = llm.call(
        system_prompt=CLARIFY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.7,
        response_format=ClarifyingQuestions,
    )
    
    questions = response.questions
    
    # Format questions for display
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    
    # Add questions as AI message
    messages.append(AIMessage(content=f"To better understand your research needs, I have a few questions:\n\n{questions_text}"))
    
    # Get user answers (interactive)
    print(f"\n{messages[-1].content}\n")
    
    answers = []
    for i, question in enumerate(questions):
        answer = input(f"Answer {i+1}: ").strip()
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
    
    # Build context from all messages
    context_parts = []
    for msg in messages:
        if msg.type == "human":
            context_parts.append(f"USER: {msg.content}")
        elif msg.type == "ai":
            context_parts.append(f"ASSISTANT: {msg.content}")
    
    context = "\n\n".join(context_parts)
    
    # Initialize OpenAI client
    llm = OpenAIClient()
    
    # Generate research brief
    user_prompt = f"""Based on the following conversation, create a comprehensive research brief.

{context}

Create a research brief that clearly outlines:
1. Research objective - what we're trying to find out
2. Key topics and subtopics to investigate
3. Expected scope and depth
4. Any specific angles or perspectives to focus on"""
    
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
        # Initial queries based on research brief
        iteration_context = ""
        num_queries = 5
        
        user_prompt = f"""Research Brief:
{research_brief}

Generate {num_queries} diverse, targeted search queries to gather comprehensive information for this research."""
    else:
        # This shouldn't happen as decide_node generates follow-up queries
        # But keeping as fallback
        compressed_findings = state.get("compressed_findings", "")
        knowledge_gaps = state.get("knowledge_gaps", [])
        
        iteration_context = f" (iteration {search_iteration + 1})"
        num_queries = 3
        
        gaps_text = "\n".join(f"- {gap}" for gap in knowledge_gaps)
        
        user_prompt = f"""Research Brief:
{research_brief}

Current Findings Summary:
{compressed_findings}

Knowledge Gaps:
{gaps_text}

Generate {num_queries} follow-up search queries to address the knowledge gaps."""
    
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
                use_autoprompt=True,
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


def reflection_node(state: ResearchState) -> ResearchState:
    """
    Reflect on findings, identify knowledge gaps, and decide if more searches needed.

    This node:
    1. Compresses all search results into a clean summary
    2. Identifies knowledge gaps based on research brief
    3. Decides if more context is needed (max 5 iterations)
    4. If yes, generates follow-up queries to address gaps
    """
    research_brief = state.get("research_brief", "")
    search_results = state.get("search_results", [])
    search_iteration = state.get("search_iteration", 0)
    
    # Initialize OpenAI client
    llm = OpenAIClient()
    
    # Format search results for analysis
    results_text = []
    for i, result in enumerate(search_results, 1):
        result_info = f"""
Result {i}:
- Query: {result.get('query', 'N/A')}
- Title: {result.get('title', 'N/A')}
- URL: {result.get('url', 'N/A')}
- Content: {result.get('text', 'N/A')[:500]}...
"""
        results_text.append(result_info)
    
    results_summary = "\n".join(results_text[:30])  # Limit to avoid token overflow
    
    # Build user prompt
    user_prompt = f"""Research Brief:
{research_brief}

Search Results from {search_iteration} iteration(s):
{results_summary}

Total results collected: {len(search_results)}

Analyze these results and:
1. Create a compressed summary of all findings
2. Identify any knowledge gaps relative to the research brief
3. Decide if more searches are needed (we're on iteration {search_iteration} of max 5)
4. If more searches needed, suggest 3-5 follow-up queries to address gaps"""
    
    # Format system prompt with iteration info
    system_prompt = DECIDE_SYSTEM_PROMPT.format(num_iterations=search_iteration)
    
    response = llm.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.5,
        response_format=DecisionOutput,
    )
    
    # Print reflection analysis
    print(f"\n=== Reflection (After Iteration {search_iteration}) ===")
    print(f"\nCompressed Findings:\n{response.compressed_findings}\n")
    
    if response.knowledge_gaps:
        print(f"Knowledge Gaps Identified:")
        for gap in response.knowledge_gaps:
            print(f"  - {gap}")
        print()
    
    print(f"Continue Searching: {response.needs_more_context}")
    
    if response.needs_more_context and response.follow_up_queries:
        print(f"\nFollow-up Queries:")
        for i, query in enumerate(response.follow_up_queries, 1):
            print(f"  {i}. {query}")
        print()
    
    return {
        "messages": state["messages"],
        "research_brief": research_brief,
        "search_queries": response.follow_up_queries if response.needs_more_context else [],
        "search_results": search_results,
        "compressed_findings": response.compressed_findings,
        "knowledge_gaps": response.knowledge_gaps,
        "search_iteration": search_iteration,
        "needs_more_context": response.needs_more_context and search_iteration < 5,
    }


def generate_report_node(state: ResearchState) -> ResearchState:
    """Generate final markdown report with citations."""
    # TODO: Implement
    pass


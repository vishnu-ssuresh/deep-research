"""Node implementations for the deep research agent."""

from langchain_core.messages import AIMessage, HumanMessage

from ..models import ClarifyingQuestions, SearchQueries
from ..prompts import (
    CLARIFY_SYSTEM_PROMPT,
    GENERATE_QUERIES_SYSTEM_PROMPT,
    RESEARCH_BRIEF_SYSTEM_PROMPT,
)
from ..services import OpenAIClient
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
    """Execute Exa searches and accumulate results."""
    # TODO: Implement
    pass


def decide_node(state: ResearchState) -> ResearchState:
    """
    Compress findings, identify knowledge gaps, and decide if more searches needed.

    This node:
    1. Compresses all search results into a clean summary
    2. Identifies knowledge gaps
    3. Decides if more context is needed
    4. If yes, generates follow-up queries
    """
    # TODO: Implement
    pass


def generate_report_node(state: ResearchState) -> ResearchState:
    """Generate final markdown report with citations."""
    # TODO: Implement
    pass


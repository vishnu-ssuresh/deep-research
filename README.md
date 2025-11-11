# Deep Research Agent

## Overview

A LangGraph-based research agent that conducts research by iteratively searching, compressing findings, and generating a detailed report. The agent is orchestrated through LangGraph, using OpenAI's LLMs and Exa's web search.

### How It Works

The agent follows a cyclic process:

1. **Clarification**: Starts by generating questions to understand the scope of the user's research topic.

2. **Research Planning**: Creates a research brief outlining objectives and key areas to investigate.

3. **Query Generation**: Generates 5 initial search queries (3 for subsequent iterations) tailored to the research brief and knowledge gaps.

4. **Parallel Data Collection**: 
   - **Web Search**: Uses Exa's search API to find and retrieve relevant web content.
   - **MCP Integration**: Uses Langchain's MCP adapters to connect with external MCP servers.

5. **Compression**: Distills accumulated search results and findings, allowing the agent to synthesize information without running into context length limits.

6. **Reflection**: Uses a "think" step to evaluate research completeness, identify knowledge gaps, and produce follow-up queries.

7. **Report Generation**: Once research is complete, it synthesizes all findings into a comprehensive report with proper citations.

8. **Export**: Saves the final report as both Markdown and PDF formats.

## Setup

### Prerequisites

- Python 3.12 or higher
- OpenAI API key
- Exa API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vishnu-ssuresh/deep-research.git
cd deep-research
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key
EXA_API_KEY=your_exa_api_key
```

## Running the Application

### Basic Usage

Run the research agent:
```bash
python agent.py
```

The agent will:
1. Ask clarifying questions about your research topic
2. Create a research brief
3. Conduct iterative searches (up to 5 iterations)
4. Generate a comprehensive report
5. Save the report as both Markdown and PDF in the `reports/` directory

## Development

### Linting

The project uses Ruff for linting and formatting:

```bash
# Check for linting issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### MCP Integration

The project includes Model Context Protocol (MCP) support for integrating external tools and data sources.

**To enable real MCP servers**:

Edit `core/helpers/nodes.py` in the `mcp_tool_node()` function:

## Project Structure

```
deep-research/
├── agent.py
├── requirements.txt
├── pyproject.toml
├── .env
│
├── core/
│   ├── __init__.py
│   │
│   ├── helpers/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── state.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── models.py
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── system_prompts.py
│   │   └── user_prompts.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── exa_client.py
│   │   └── openai_client.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── report_utils.py
│   │
│   └── exceptions.py
│
└── reports/
```

### Customization

**Adjust iteration count**: Edit `should_continue_searching()` in `core/helpers/graph.py`

**Modify search parameters**: Edit `search_node()` in `core/helpers/nodes.py`

**Change LLM parameters**: Edit `core/services/openai_client.py`

**Configure MCP servers**: Edit `server_configs` in `mcp_tool_node()` in `core/helpers/nodes.py`

## Next Steps

1. **CLI Configuration**: Extend `agent.py` to support command-line arguments like `--max-iterations` and `--model`, allowing the user to customize behavior without directly modifying the code.

2. **Human-in-the-Loop Mode**: Add an interactive mode where the agent pauses at each iteration to display the reflection node's thought process and allow the user to steer the research direction.

3. **Visual Enhancements**: Add image and chart generation through MCP integration by connecting to image search APIs or chart/diagram generation tools to visually enhance the final report.

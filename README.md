# Deep Research

A LangGraph-based research agent that conducts comprehensive research by iteratively searching, compressing findings, and generating detailed reports.

## Overview

Deep Research is an AI-powered research assistant that automates the process of gathering, analyzing, and synthesizing information from multiple sources. The system uses a graph-based workflow to:

- Clarify research objectives through interactive questioning
- Generate targeted search queries
- Execute parallel web searches and MCP tool integrations
- Compress and analyze findings
- Iteratively refine research through reflection
- Generate comprehensive reports in Markdown and PDF formats

The agent leverages OpenAI's GPT models for reasoning and Exa for web search capabilities. It includes support for Model Context Protocol (MCP) servers to integrate external tools and data sources.

## Setup

### Prerequisites

- Python 3.12 or higher
- OpenAI API key
- Exa API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-research.git
cd deep-research
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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

### Example Workflow

```bash
$ python agent.py

# The agent will prompt you for:
# - Initial research query
# - Answers to clarifying questions
# - The agent then runs autonomously

# Output will be saved to:
# reports/your-research-topic.md
# reports/your-research-topic.pdf
```

## Development

### Linting

The project uses Ruff for linting and formatting:

```bash
# Check for linting issues
./venv/bin/ruff check .

# Auto-fix issues
./venv/bin/ruff check --fix .

# Format code
./venv/bin/ruff format .
```

### MCP Integration

The project includes Model Context Protocol (MCP) support for integrating external tools and data sources.

**Current Status**: MCP node runs in dummy mode by default (no external dependencies required).

**To enable real MCP servers**:

Edit `core/helpers/nodes.py` in the `mcp_tool_node()` function:

```python
server_configs = {
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/data"],
    },
}
```

Supported MCP server types:
- `stdio`: Local subprocess communication (e.g., filesystem, SQLite)
- `streamable_http`: HTTP-based remote servers

### Architecture

The system uses LangGraph to orchestrate a stateful workflow:

```
START → clarify → research_brief → generate_queries
         ↓
    [search, mcp_tools] (parallel)
         ↓
    compress → reflect → (loop or continue) → generate_report → save_pdf → END
```

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

**Change LLM models**: Edit model parameters in `core/services/openai_client.py`

**Configure MCP servers**: Edit `server_configs` in `mcp_tool_node()` in `core/helpers/nodes.py`

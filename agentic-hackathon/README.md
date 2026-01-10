# Agentic AI Hackathon Scaffold

Deployment-ready scaffold for agentic workflows with LangGraph orchestration, OpenAI LLMs, FAISS RAG, MCP stubs, and an AutoGen-style multi-agent harness. This template is designed for hackathons: fast to extend, opinionated defaults, and clean separation of planner/executor/verifier.

## What''s Included
- LangGraph workflow: planner -> retrieve -> act -> verify
- RAG pipeline: FAISS + OpenAI embeddings + retriever tool (configurable via env)
- MCP template: registry, client, and tool adapter stubs
- AutoGen-style harness: multi-agent orchestrator with role configs
- DeepAgents-style harness: multi-step execution, subagent-friendly
- Tool stubs: web search (optional), internal API, browser, custom
- PDF/doc ingestion script for building FAISS index
- Containerized build via Docker

## Quickstart
1) Create env file
```
copy .env.example .env
```
2) Install deps
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
3) Ingest docs
```
python -m src.tools.ingest --input ./docs --index ./data/faiss_index
```
4) Run agent
```
python -m src.agent_core "Plan a project roadmap from the docs"
```

## MCP Template Quickstart
1) Register MCP servers (comma-separated `name=url` pairs)
```
MCP_SERVERS=files=http://localhost:9000,search=http://localhost:9001
```
2) Call a tool via the executor tool list (inside your agent/tooling flow)
```
call_mcp_tool("files", "list", {"path": "/"})
```

## AutoGen-Style Multi-Agent Quickstart
Use the orchestrator to run round-robin agent stubs, then replace `default_responder`
with real LLM calls or tool execution.
```
from src.autogen import AgentConfig, MultiAgentOrchestrator

agents = [
    AgentConfig(name="Planner", role="planner", system_prompt="Break down tasks."),
    AgentConfig(name="Builder", role="executor", system_prompt="Implement steps."),
]
team = MultiAgentOrchestrator(agents)
messages = team.run("Draft a RAG integration plan.", rounds=2)
```


## Docker
```
docker build -t agentic-hackathon .
docker run --env-file .env agentic-hackathon
```

## Project Structure
```
agentic-hackathon/
+-- .env.example
+-- Dockerfile
+-- requirements.txt
+-- src/
   +-- __init__.py
   +-- agent_core.py
   +-- planner.py
   +-- executor.py
   +-- verifier.py
   +-- deepagents_harness.py
   +-- tools/
      +-- __init__.py
      +-- rag_tool.py
      +-- ingest.py
      +-- search_tool.py
      +-- browser_tool.py
      +-- db_tool.py
      +-- custom_tool.py
   +-- utils/
       +-- logging.py
       +-- schemas.py
       +-- tracing.py
+-- tests/
+-- notebooks/
```

## Notes on Research-Inspired Patterns
- Planning/execution separation mirrors planner-executor agents.
- Verifier loop encourages reflective correction.
- Context engineering: retriever tool only injects relevant chunks.
- Extensible harness supports role-based or multi-agent patterns.

## Extending
- Add new tools in `src/tools/` and register them in `src/executor.py`.
- Extend MCP behaviors in `src/mcp/` and swap transports to real MCP clients.
- Replace `default_responder` in `src/autogen/multi_agent.py` with LLM-driven logic.
- Swap models via `OPENAI_MODEL` or use `EMBEDDING_MODEL` in `.env`.
- Add memory strategies (summaries, episodic notes) in `src/agent_core.py`.
# Agentic AI Hackathon Scaffold

Deployment-ready scaffold for agentic workflows with LangGraph orchestration, OpenAI LLMs, FAISS RAG, and a DeepAgents-style harness. This template is designed for hackathons: fast to extend, opinionated defaults, and clean separation of planner/executor/verifier.

## What''s Included
- LangGraph workflow: planner -> retrieve -> act -> verify
- RAG pipeline: FAISS + OpenAI embeddings + retriever tool
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
¦   +-- __init__.py
¦   +-- agent_core.py
¦   +-- planner.py
¦   +-- executor.py
¦   +-- verifier.py
¦   +-- deepagents_harness.py
¦   +-- tools/
¦   ¦   +-- __init__.py
¦   ¦   +-- rag_tool.py
¦   ¦   +-- ingest.py
¦   ¦   +-- search_tool.py
¦   ¦   +-- browser_tool.py
¦   ¦   +-- db_tool.py
¦   ¦   +-- custom_tool.py
¦   +-- utils/
¦       +-- logging.py
¦       +-- schemas.py
¦       +-- tracing.py
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
- Swap models via `OPENAI_MODEL` or use `EMBEDDING_MODEL` in `.env`.
- Add memory strategies (summaries, episodic notes) in `src/agent_core.py`.
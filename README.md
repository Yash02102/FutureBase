# Deep Agents Commerce Scaffold

Clean LangGraph + Deep Agents architecture with MCP tools, RAG, optional specialist subagents, and token-efficient memory.

## Quickstart
1) Create env file
```
copy .env.example .env
```
2) Install deps
```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
3) Start MCP server
```
py -3.12 -m src.mcp.server
```
4) (Optional) Build RAG index
```
py -3.12 -m src.rag_build --input ./docs --index ./data/faiss_index
```
5) Run the agent
```
py -3.12 -m src.app --session-id demo "Buy me a wireless headset under 5000"
```

## Chat UI
Start the multi-turn chat interface with FastAPI:
```
py -3.12 -m uvicorn src.web:app --reload
```
Open `http://localhost:8000` in a browser. The UI stores the session id locally to keep the conversation thread.

## Key Environment Options
- `LLM_PROVIDER=openai|anthropic|openai_compatible`
- `LLM_MODEL=gpt-4o-mini` (or Anthropic model)
- `LLM_API_KEY` and provider-specific keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
- `EMBEDDING_PROVIDER=openai` and `EMBEDDING_MODEL=text-embedding-3-small`
- `RAG_ENABLED=true|false`, `RAG_INDEX_PATH=./data/faiss_index`
- `RAG_VECTOR_K=6`, `RAG_BM25_K=6`, `RAG_REWRITE_COUNT=2`, `RAG_RERANK_TOP_K=6`
- `MCP_SERVER_URL=http://localhost:8000/mcp`
- `FILESYSTEM_ROOT=C:\\abs\\path` to enable Deep Agents filesystem tools
- `SUBAGENTS_ENABLED=true|false` to attach specialist subagents to the main agent
- `MEMORY_BACKEND=ephemeral|filesystem`, `MEMORY_PATH=./data/memory`
- `HITL_MODE=auto|manual`, `HITL_TOOLS=checkout,cart_add`
- `CACHE_MODE=memory|off`

## MCP Server
The MCP server provides mock commerce APIs (catalog, inventory, pricing, cart, checkout, support).
By default it runs with streamable HTTP at `http://localhost:8000/mcp`.

## Notes
- The main agent uses generic read-only tools and can delegate specialized actions (cart, checkout, returns) to subagents via the task tool. HITL gates sensitive actions.
- HITL manual mode prompts in the CLI. It uses LangGraph checkpoints and deep agents middleware.
- RAG only activates if `RAG_ENABLED=true` and the index exists.

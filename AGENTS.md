# AGENTS.md

Run: `uv run python main.py` (Python 3.12, `uv` as the build/resolver; `.venv` is managed by uv). Do not use `python -m multi_agent_ai_wedding_planner` or `multi-agent-ai-wedding-planner` — the `src/.../__init__.py` `main` is a placeholder and the real runner is `main.py`.

## Environment

- `main.py` requires a local omni route proxy on `http://localhost:20128/v1` with `OMNIROUTE_API_KEY` in the environment; set these in a local `.env`. `.env` and `.venv` are gitignored.
- `TAVILY_API_KEY` is required for `tools.web_search` (`TavilyClient()`).
- `models.init_chat_model` uses `model="free_combo"`, provider `openai`, plus `base_url` and `api_key`.

## Run

- Main flow: `uv run python main.py` — reads requirements from stdin, builds `WeddingPlannerAgent` with tools `delegate_subagent1` / `delegate_subagent2` that invoke two LangGraph/LangChain subagents, each tool'd with `web_search`.

## Gotchas

- `tools.py` imports `load_dotenv` but never calls it (`load_dotenv` line is a no-op). This is a bug: add `load_dotenv()`.
- `test.py` is not a test harness — it calls the LangSearch API with a hardcoded bearer token. Do not treat it as a verification script; it ships an embedded credential.
- `requirements.txt` is stale; `pyproject.toml` + `uv.lock` are the source of truth.
- There is no lint/typecheck/test/CI config in the repo. Do not assume any `pytest`, `ruff`, or pre-commit hooks exist.

## Files

- `main.py` — entrypoint flow.
- `agent.py` — subagents and the two delegation tools.
- `tools.py` — `web_search` (Tavily).
- `prompts.py` — system/user prompts.
- `models.py` — model init.

# Multi-Agent AI Wedding Planner

A multi-agent AI system that turns a structured wedding brief into a client-ready planning plan — venue and vendor research, budget allocation, timeline, and risk analysis — using LangChain/LangGraph agents with live web search.

Two ways to use it: a polished **Streamlit UI** (`app.py`) or a **CLI** (`main.py`).

## Features

- **Multi-agent architecture** — a main planner agent delegates research to two sub-agents, each equipped with Tavily web search, then synthesizes their findings into one plan
- **Structured wedding brief** — couple, location, date, guest count, budget range, style direction, priorities, must-haves, constraints, and cultural/ceremonial details
- **Any LLM provider** — bring your own key: OpenAI, Anthropic, Groq, or an OmniRoute-compatible proxy (first key found wins)
- **Streamlit dashboard** — animated glassmorphism UI with live summary cards, a planning-lens sidebar, run metrics, saved runs, and Markdown export
- **Live market research** — sub-agents pull current venue/vendor/logistics data via Tavily before synthesis

## Architecture

```
Streamlit UI / CLI
    -> structured wedding requirements
    -> WEDDING_PLANNER_AGENT_PROMPT
    -> MainWeddingPlannerAgent
        -> delegate_subagent1(query)
            -> SubAgent1 -> Tavily web_search
        -> delegate_subagent2(query)
            -> SubAgent2 -> Tavily web_search
    -> synthesized wedding plan (Markdown)
```

## Tech Stack

| Layer | Tools |
|---|---|
| Agents | LangChain 1.x, LangGraph |
| Search | Tavily API |
| UI | Streamlit |
| Models | OpenAI, Anthropic, Groq, or OmniRoute proxy (OpenAI-compatible) |
| Runtime | Python 3.12, uv |

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A Tavily API key (required — https://tavily.com)
- At least one LLM provider key (OpenAI, Anthropic, Groq) **or** a running OmniRoute-compatible proxy

## Setup

```bash
git clone https://github.com/hamzawakili15-cpu/Multi-Agent-AI-Wedding-Planner.git
cd Multi-Agent-AI-Wedding-Planner
uv sync
```

Copy the environment template and fill in your keys:

```bash
cp .env.example .env
```

```env
# At least one LLM provider key is required
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GROQ_API_KEY=
OMNIROUTE_API_KEY=

# Search (required)
TAVILY_API_KEY=
```

## Run

### Streamlit UI (recommended)

```bash
uv run streamlit run app.py
```

Opens at http://localhost:8501. You can also paste API keys directly in the sidebar — they override `.env` for the session.

### CLI

```bash
uv run python main.py
```

Prompts for your wedding requirements on stdin, runs the agent chain, and prints the plan.

## How It Works

1. **Brief** — you fill in the wedding brief (couple, budget, style, priorities, constraints).
2. **Prompt build** — the brief is formatted into `WEDDING_PLANNER_AGENT_PROMPT`.
3. **Delegation** — the main planner splits research into focused queries and delegates them to SubAgent1 and SubAgent2 via tools.
4. **Research** — each sub-agent runs Tavily web searches for current market data (venues, vendors, logistics).
5. **Synthesis** — the main agent merges the research into an executive summary, venue/vendor recommendations with rationale, budget by category, a now-to-wedding-day timeline, and risks/tradeoffs/open questions.
6. **Export** — the result is shown in the UI with runtime metrics and downloadable as Markdown.

## Project Structure

```
.
├── app.py        # Streamlit UI (dashboard, brief form, results)
├── main.py       # CLI entrypoint
├── agent.py      # SubAgents + delegation tools
├── models.py     # Multi-provider model factory
├── tools.py      # Tavily web_search tool
├── prompts.py    # System + user prompts
└── src/multi_agent_ai_wedding_planner/  # package placeholder
```

## Configuration Notes

- **Provider priority** — `models.py` picks the first key found in this order: `OPENAI_API_KEY` → `ANTHROPIC_API_KEY` → `GROQ_API_KEY` → `OMNIROUTE_API_KEY`.
- **OmniRoute** — used with `base_url=http://localhost:20128/v1` and model `free_combo`; requires a local OpenAI-compatible proxy.
- **Default models** — `gpt-5-nano` (OpenAI), `claude-haiku-4-5` (Anthropic), `llama-3.3-70b-versatile` (Groq); edit `PROVIDER_CONFIGS` in `models.py` to change them.
- **Key hygiene** — `.env` is gitignored; never commit real keys.

## License

See repository for license details.

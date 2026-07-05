# Modular RAG Template

A **fork-and-go RAG chatbot** that answers questions about *your* organization,
grounded in its own web pages. Runs **fully local by default** (Ollama + Chroma) —
no API keys, no data leaving your machine — and is **provider-agnostic**, so you
can swap in OpenAI, Anthropic, or pgvector by changing one config value.

Built as a reusable architecture project: clone it, point it at your URLs, and
you have a grounded, cited chatbot. See [RAG_PLAN.md](RAG_PLAN.md) for the full
architecture and the *why* behind each component.

## Features

- **Local-first** — Qwen via Ollama + Chroma out of the box; zero cloud cost.
- **Provider-agnostic** — LLM, embeddings, and vector store are chosen from
  config ([rag/providers.py](rag/providers.py)). Ollama/OpenAI/Anthropic +
  Chroma/pgvector.
- **Modular RAG** — every advanced feature is a toggle: hybrid search,
  reranking, multi-query fusion, Small2Big chunking, Reverse-HyDE, CRAG.
- **Grounded + cited** — answers only from your sources, with source URLs, and
  an "I don't know" guardrail.
- **Batteries included** — crawler (respects `robots.txt`), CLI + Streamlit UIs,
  and a RAGAS evaluation harness.

## Quickstart

```bash
./setup.sh                      # venv + deps + pull models + seed config
# edit .env  -> set ORGANIZATION_NAME
# edit sources.txt -> your page URLs
make ingest                     # crawl + build the index
make chat                       # ask questions in the terminal
```

Prefer a browser UI? `make ui` (Streamlit). Prefer manual steps? See below.

## Manual setup

1. **Ollama** running locally (for the default provider):
   ```bash
   ollama pull qwen2.5:0.5b-instruct
   ollama pull nomic-embed-text
   ```
   Or use a cloud provider instead — see *Swapping providers*.
2. **Python 3.11+** deps: `pip install -r requirements.txt`
3. **Configure**: copy `.env.example` → `.env`, set `ORGANIZATION_NAME`.
4. **Sources**: copy `sources.txt.example` → `sources.txt`, add your URLs.
5. **Build + run**: `python -m ingest.build_index` then `python cli.py`.

## Swapping providers

Everything routes through [rag/providers.py](rag/providers.py). Change these in
`.env` (and install the matching optional package from `requirements.txt`):

| Goal | Set |
|------|-----|
| OpenAI for generation | `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4o-mini`, `OPENAI_API_KEY=…` |
| OpenAI embeddings | `EMBEDDING_PROVIDER=openai`, `EMBEDDING_MODEL=text-embedding-3-small` |
| Anthropic generation | `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-…`, `ANTHROPIC_API_KEY=…` |
| Postgres/pgvector store | `VECTORSTORE_PROVIDER=pgvector`, `PGVECTOR_CONNECTION=…` |

> Changing `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL` or the chunk sizes requires a
> re-ingest (`make ingest`) — embeddings must be produced by the same model used
> at query time.

## How it fits together

```
sources.txt ──> ingest/ ──> chroma_db/ + docstore/ <── rag/ <── cli.py / app.py
                 (writes)                            (reads)
```

- `ingest/` — fills the stores (crawl → chunk → enrich → build_index).
- `rag/` — reads them per message (expand → retrieve → generate).
- `config.py` — every model choice and feature toggle, in one place.
- `rag/providers.py` — the only file that knows about concrete providers.

## Configuration

All settings live in [config.py](config.py) and are overridable via env / `.env`.
Key ones:

| Variable | Purpose | Default |
|----------|---------|---------|
| `ORGANIZATION_NAME` | Who the bot represents | `Your Organization` |
| `ASSISTANT_DESCRIPTION` | Persona line in the system prompt | derived from name |
| `LLM_PROVIDER` / `EMBEDDING_PROVIDER` / `VECTORSTORE_PROVIDER` | Engines | `ollama` / `ollama` / `chroma` |
| `LLM_MODEL` / `EMBEDDING_MODEL` | Model names | `qwen2.5:0.5b-instruct` / `nomic-embed-text` |
| `USE_HYBRID_SEARCH` / `USE_RERANKER` / `USE_MULTI_QUERY` / `USE_REVERSE_HYDE` | Feature toggles | `true` |
| `USE_CRAG` | Self-grading (needs 7b+/cloud) | `false` |
| `CRAWL_MAX_DEPTH` / `RESPECT_ROBOTS_TXT` | Crawl scope + politeness | `1` / `true` |

## Make targets

`make setup` · `make ingest` · `make chat` · `make ui` · `make eval` · `make clean`

## License

MIT — see [LICENSE](LICENSE). Fork it, adapt it, ship it.

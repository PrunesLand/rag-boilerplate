# Modular RAG Template

A reusable Retrieval Augmentation Generation (RAG) Architecture template complete with MLOps capabilities with LangChain. As businesses continue to take further interest in the deployment of Large Language Models (LLMs) in their business practices, there is a surging need for quick and quality deployment of such project. The aim of this repository is to provide organizations with a template of an end-to-end question and answer chatbot with interchangable components as according to technical requirements.

## Contents

- [Quickstart](#quickstart)
- [Manual setup](#manual-setup)
- [Swapping providers](#swapping-providers)
- [Architecture overview](#architecture-overview)
- [Configuration](#configuration)
  - [Key settings](#key-settings)
- [Make targets](#make-targets)
- [References](#references)
- [License](#license)

## Quickstart

```bash
./setup.sh                      # create venv, install deps, seed config files
# provision your model backend — see Manual setup > Model backend below
# edit config.py to set ORGANIZATION_NAME and any other settings
# create sources.txt and list your page URLs, one per line
make ingest                     # crawl sources and build the index
make chat                       # query the assistant in the terminal
```

A browser interface is available via `make ui` (Streamlit). Manual setup steps
are described below.

## Manual setup

1. **Model backend.** Provision the provider you intend to use for the LLM and
   embeddings, then select it in [config.py](config.py). See
   [Swapping providers](#swapping-providers) for the full list.
   - *Local (default)* — install [Ollama](https://ollama.com) and pull the
     models named in your config:
     ```bash
     ollama pull qwen2.5:0.5b-instruct
     ollama pull nomic-embed-text
     ```
   - *Cloud* — obtain an API key for your chosen provider (e.g. OpenAI,
     Anthropic); no local models are required. Keep the key as a secret — see
     step 3.
2. **Python 3.11+** dependencies: `pip install -r requirements.txt` (install any
   optional provider package listed in `requirements.txt` for your backend).
3. **Configuration.** How the project runs — provider selection, model names,
   feature toggles, crawl behavior, and everything else in
   [Key settings](#key-settings) — is set by editing [config.py](config.py)
   directly.

   Secrets (API keys, database connection strings) do not belong in
   `config.py`. Supply them as environment variables instead: copy
   `.env.example` to `.env`, fill in only the secrets your chosen providers
   need, then load it into your shell before running anything:
   ```bash
   cp .env.example .env               # then fill in your secret values
   set -a && source .env && set +a    # export every variable into the current shell
   ```
   Repeat the `source` step in every new shell session (or add it to your shell
   profile; [`direnv`](https://direnv.net/) can automate this per directory).
   The `make` targets invoke Python directly, so source `.env` in the same
   shell beforehand.
4. **Sources.** Create a file named `sources.txt` in the project root and list
   your source URLs, one per line (see `sources.txt.example` for the format).
5. **Build and run**: `python -m ingest.build_index`, then `python cli.py`.

## Swapping providers

All provider selection is routed through [rag/providers.py](rag/providers.py).
Each of the three roles — generation, embeddings, and vector store — is chosen
independently. Set the variables for your choice in the environment. The base
LangChain stack is installed during setup (`pip install -r requirements.txt`);
provider-specific packages such as `langchain-openai`, `langchain-anthropic`, and
`langchain-postgres` are listed there as optional entries — uncomment the one for
your provider before installing, or install it directly.

**Generation:** 

| Provider | Configuration |
|----------|---------------|
| Ollama | `LLM_PROVIDER=ollama`, `LLM_MODEL=qwen2.5:0.5b-instruct` |
| OpenAI | `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4o-mini`, `OPENAI_API_KEY=…` |
| Anthropic | `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-…`, `ANTHROPIC_API_KEY=…` |

**Embeddings:** 

| Provider | Configuration |
|----------|---------------|
| Ollama | `EMBEDDING_PROVIDER=ollama`, `EMBEDDING_MODEL=nomic-embed-text` |
| OpenAI | `EMBEDDING_PROVIDER=openai`, `EMBEDDING_MODEL=text-embedding-3-small` |

**Vector store:** 

| Provider | Configuration |
|----------|---------------|
| Chroma | `VECTORSTORE_PROVIDER=chroma` (local, persisted to `chroma_db/`) |
| pgvector | `VECTORSTORE_PROVIDER=pgvector`, `PGVECTOR_CONNECTION=…` |

> Changing `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, or the chunk sizes requires
> re-ingestion (`make ingest`): embeddings must be produced by the same model
> used at query time.

## Architecture overview

```
sources.txt ──> ingest/ ──> vector store + docstore/ <── rag/ <── cli.py / app.py
                 (writes)                            (reads)
```

- `ingest/` — populates the stores (crawl → chunk → enrich → build_index).
- `rag/` — reads the stores per request (expand → retrieve → generate).
- `config.py` — centralizes every model choice and feature toggle.
- `rag/providers.py` — the sole module aware of concrete providers.
- **Vector store** — holds the embedded child chunks; backend is chosen via
  `VECTORSTORE_PROVIDER` (Chroma persists locally to `chroma_db/`; pgvector
  lives in your Postgres instance). See [Swapping providers](#swapping-providers).
- **`docstore/`** — holds the parent chunks for Small2Big expansion; always a
  local, fixed implementation regardless of `VECTORSTORE_PROVIDER`.

## Configuration

All behavioral settings live in [config.py](config.py) — edit it directly to
change how the project runs. Secrets (API keys, connection strings) stay out of
`config.py` and are supplied via environment variables instead; see
[Manual setup](#manual-setup) step 3.

### Key settings

| Variable | Purpose | Default |
|----------|---------|---------|
| `ORGANIZATION_NAME` | Organization the assistant represents | `Your Organization` |
| `ASSISTANT_DESCRIPTION` | Persona line in the system prompt | derived from name |
| `LLM_PROVIDER` / `EMBEDDING_PROVIDER` / `VECTORSTORE_PROVIDER` | Engines | `ollama` / `ollama` / `chroma` |
| `LLM_MODEL` / `EMBEDDING_MODEL` | Model names | `qwen2.5:0.5b-instruct` / `nomic-embed-text` |
| `USE_HYBRID_SEARCH` / `USE_RERANKER` / `USE_MULTI_QUERY` / `USE_REVERSE_HYDE` | Feature toggles | `true` |
| `USE_CRAG` | Self-grading (requires a 7B+ or cloud model) | `false` |
| `CRAWL_MAX_DEPTH` / `RESPECT_ROBOTS_TXT` | Crawl scope and politeness | `1` / `true` |

## Make targets

`make setup` · `make ingest` · `make chat` · `make ui` · `make eval` · `make clean`

## References

This project draws on the following surveys on Retrieval-Augmented Generation (RAG):

- Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., & Wang, H. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997. https://arxiv.org/abs/2312.10997

- Gao, Y., Xiong, Y., Wang, M., & Wang, H. (2024). *Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks*. arXiv:2407.21059. https://arxiv.org/abs/2407.21059

## License

Released under the MIT License; see [LICENSE](LICENSE).

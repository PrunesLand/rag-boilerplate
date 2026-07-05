# University Chatbot — Modular RAG Architecture Plan (v2)

A Retrieval-Augmented Generation (RAG) chatbot for answering questions about the
university, grounded in its public webpages. Runs locally on a laptop using a
Qwen model served via **Ollama**, orchestrated with **LangChain**, and traced
with **LangSmith**.

> **What changed in v2:** the design moves from a linear *Advanced RAG* chain to
> a *Modular RAG* architecture. Retrieval is no longer one-shot — queries are
> expanded and self-checked, and each stage is an independently swappable module
> toggled from `config.py`. New modules: multi-query expansion (RAG Fusion), a
> CRAG relevance check, Small2Big chunking, Reverse-HyDE metadata, and a RAGAS
> evaluation harness.

---

## 1. Mental Model: Two Separate Pipelines

The system splits into two independent halves. Keep this separation clear — it is
the backbone of the design.

- **Ingestion (offline, batch)** — runs occasionally, when pages change.
- **Query (online, per-message)** — runs every time a user asks something.

```
sources.txt ──> ingest/ ──> chroma_db/ <── rag/ <── app.py
                  (writes)            (reads)
```

The mental model in one line: **Ingestion fills the vector DB; Query reads from
it.** The vector DB (`chroma_db/`) is the single handoff point between the two
halves.

New in v2: ingestion now does more work up front (generating questions and
summaries per chunk) so the query pipeline can stay fast. The principle holds —
**push expensive, reusable work into ingestion; keep the online path lean.**

---

## 2. Ingestion Pipeline (offline)

```
URLs → Crawl → Clean → Chunk (Small2Big) → Enrich → Embed → Store in Vector DB
```

| Stage  | Detail |
|--------|--------|
| Crawl  | `RecursiveUrlLoader` over the university domain, scoped tightly (`max_depth=2`, domain filter, exclude `/login`, calendars, unwanted PDFs). |
| Clean  | Strip nav/footer/ads with `trafilatura` (automatic main-content extraction). Keep main content + metadata (`url`, `title`, heading path). |
| Chunk  | **Small2Big**: split into small ~200-token child chunks for precise matching, each linked to a larger ~800-token parent chunk returned to the LLM. (`ParentDocumentRetriever`.) |
| Enrich | Per chunk, generate (a) 2–3 **hypothetical questions** it answers — *Reverse HyDE*; (b) a one-line **summary**. Store both as metadata. |
| Embed  | `OllamaEmbeddings` (`nomic-embed-text`). Embed the child chunks **and** the hypothetical questions. |
| Store  | Two stores: child embeddings → `Chroma(persist_directory="./chroma_db")`; parent chunks → a persistent docstore (`./docstore`). `ParentDocumentRetriever` wires them together. Both persisted to disk. |

Re-run this whenever pages change. Use a content hash to skip unchanged pages.

### Why Small2Big

Fixed 600-token chunks force a tradeoff: small chunks retrieve precisely but lack
context; large chunks carry context but retrieve noisily. Small2Big sidesteps it —
match on the small chunk, hand the LLM the big one.

```
Match on:   [200-token child]  ──points to──>  [800-token parent]
Send to LLM:                                    [800-token parent]
```

**Two stores, not one.** This is the detail that breaks builds if missed:
`ParentDocumentRetriever` keeps the child embeddings in Chroma (for matching) but
the parent chunks in a *separate* docstore (for lookup). The parents do **not**
live in the vector DB. If you don't give it a persistent docstore, it silently
uses an in-memory one and your parents disappear on restart — retrieval then
matches children but has nothing to expand them to. Point the docstore at
`./docstore` so it survives across runs, and build both stores together in
`rag/store.py` so the ingestion and query sides wire them identically.

### Why Reverse HyDE

User questions and source prose rarely share vocabulary. By generating the
questions each chunk *answers* at ingestion time and embedding those, retrieval
matches question-to-question instead of question-to-prose — closing the semantic
gap. Because it runs offline, it adds **zero query-time latency**.

```python
# during build_index.py, per chunk
hypothetical_questions = llm.invoke(
    f"Generate 3 questions this text answers:\n{chunk.page_content}"
)
chunk.metadata["hypothetical_questions"] = hypothetical_questions
chunk.metadata["summary"] = llm.invoke(f"One-line summary:\n{chunk.page_content}")
```

---

## 3. Query Pipeline (online)

```
Question → Rewrite → Expand → Hybrid Retrieve + RRF → CRAG Check
        → Rerank → Small2Big Expand → Build Prompt → Qwen → Answer + Citations
```

| Stage     | Detail |
|-----------|--------|
| Rewrite   | History-aware: turn follow-up questions into standalone queries. |
| Expand    | **Multi-query / RAG Fusion**: generate 3–5 query variants, retrieve each in parallel. |
| Retrieve  | **Hybrid (default)**: vector + `BM25Retriever` via `EnsembleRetriever`, top-k (15–20) per variant. |
| Merge     | **Reciprocal Rank Fusion (RRF)** combines the variant result lists into one ranking. |
| CRAG      | Qwen scores top-chunk relevance: **HIGH** → proceed; **LOW** → reformulate & retry once; **NONE** → "I don't know". |
| Rerank    | Cross-encoder narrows merged pool → top-5. |
| Expand    | Swap matched child chunks for their **parent** chunks (Small2Big). |
| Prompt    | "Answer ONLY from context, cite source URLs" template. |
| Generate  | `ChatOllama` running Qwen (`temperature=0`). |
| Guardrail | The CRAG **NONE** branch and low rerank scores both trigger the "I don't have that info" reply. |

### Prompt Skeleton

```
You are a helpful assistant for [University Name]. Answer ONLY using the
context below. If the answer isn't in the context, say you don't know and
suggest where to look. Always cite the source URL(s).

Context:
{retrieved_chunks_with_urls}

Question: {user_question}
```

### UI contract (`cli.py` and `app.py`)

Both front-ends stay thin by talking to `rag/` through one small interface, so
neither knows anything about retrieval internals:

```python
retriever = build_retriever()                  # from rag/store.py; built once
result = answer(query, history, retriever)      # from rag/generate.py
    result.source_urls   # list[str] — resolved after retrieval, before streaming
    result.tokens        # Iterator[str] — yields answer tokens as they generate
```

`answer()` runs the whole query pipeline internally (rewrite → expand → hybrid +
RRF → CRAG → rerank → parent expansion) and returns once generation is ready to
stream, with sources already resolved. Because the retrieval/CRAG phase produces
no visible output, a front-end shows a status indicator during it, then streams
tokens once generation starts — so a CRAG retry reads as "still working," not a
hang. Sources render beneath the answer. Keeping this contract stable is what lets
`cli.py` today and `app.py` later share the exact same pipeline unchanged.
- **CRAG relevance check** — a lightweight self-grade *before* generation. Catches
  the case where retrieval returned something but it's off-topic, and retries or
  bails instead of hallucinating over bad context.
- **Small2Big expansion** — the precision/context win from §2, realised at query
  time by returning parents.

---

## 4. Component Map (Ollama + LangChain)

| Pipeline step      | LangChain piece |
|--------------------|-----------------|
| Load web pages     | `WebBaseLoader` / `RecursiveUrlLoader` |
| Clean HTML         | `trafilatura` |
| Chunk (Small2Big)  | `RecursiveCharacterTextSplitter` (parent + child) via `ParentDocumentRetriever` |
| Enrich (questions/summary) | LCEL chain calling `ChatOllama` per chunk |
| Embed + store      | `OllamaEmbeddings` → `Chroma` |
| Expand queries     | `MultiQueryRetriever` (+ RRF merge) |
| Hybrid retrieve    | `EnsembleRetriever` = `BM25Retriever` + vector retriever **(default on)** |
| CRAG check         | LCEL grader chain → branch (`RunnableBranch`) |
| Rerank             | `ContextualCompressionRetriever` + cross-encoder |
| Parent expansion   | `ParentDocumentRetriever` |
| Generate           | LCEL chain: `prompt \| ChatOllama \| StrOutputParser` |
| Multi-turn         | `create_history_aware_retriever` + `RunnableWithMessageHistory` |
| Evaluation         | RAGAS over a fixed test set |
| Tracing/debug      | LangSmith (`LANGCHAIN_TRACING_V2=true`) |

### Models (via Ollama)
- **LLM**: `qwen2.5:0.5b-instruct` (smallest Qwen; swap up to 7b for quality)
- **Embeddings**: `nomic-embed-text`

---

## 5. Project Layout

```
university_llm/
├── config.py            # central settings: model, embedding, tuning knobs
├── requirements.txt     # pinned dependencies (see §6)
├── .env                 # OLLAMA_BASE_URL, LANGSMITH_* etc. (gitignored)
├── .gitignore           # chroma_db/, docstore/, .env, __pycache__
├── README.md            # run order: ingest → cli/app; how to configure
├── ingest/
│   ├── __init__.py
│   ├── crawl.py         # fetch + extract clean text (+ url/title metadata)
│   ├── chunk.py         # Small2Big split into parent/child
│   ├── enrich.py        # hypothetical questions + summary → chunk metadata
│   └── build_index.py   # embed children → Chroma; write parents → docstore
├── rag/
│   ├── __init__.py
│   ├── store.py         # build + return Chroma vectorstore + docstore + retriever
│   ├── expand.py        # multi-query generation + RRF merge
│   ├── retrieve.py      # hybrid search, CRAG check, rerank, parent expansion
│   └── generate.py      # build prompt, call Qwen via Ollama
├── eval/
│   ├── __init__.py
│   ├── testset.jsonl    # ~30 ground-truth Q/A pairs
│   └── run_ragas.py     # score the pipeline, log to LangSmith
├── cli.py               # terminal chat UI (dev loop; runs before app.py exists)
├── app.py               # Streamlit / FastAPI chat UI
├── sources.txt          # seed URLs
├── chroma_db/           # persisted child-vector store (gitignored)
└── docstore/            # persisted parent chunks for Small2Big (gitignored)
```

### Folder responsibilities

- **`ingest/`** — fills the stores. Run as a script, not during chat.
  Flow: `crawl.py` → `chunk.py` → `enrich.py` → `build_index.py`.
- **`rag/`** — reads from the stores. Runs on every user message; never
  crawls or embeds. Flow: `expand.py` → `retrieve.py` → `generate.py`.
- **`rag/store.py`** — the single place that knows how to connect to the stores.
  Builds and returns the Chroma vectorstore, the parent docstore, and the
  assembled retriever from `config.py`. Imported by `build_index.py` (to write),
  and by `retrieve.py` / `eval/run_ragas.py` (to read). Keeps store-wiring in one
  place — the same instinct that put all settings in `config.py`.
- **`eval/`** — offline quality harness. Depends on `rag/`, never touched during
  normal chat. Run it after every change to retrieval or chunking.
- **`cli.py`** — terminal chat UI: the same thin shell as `app.py` but for the
  command line. Holds the chat loop + session history (a plain list — no rerun
  model, so none of Streamlit's caching/`session_state` gymnastics). Calls into
  `rag/` through the same `answer()` / `build_retriever()` contract `app.py` uses,
  so both share one interface. Runs in a stub echo mode until `rag/` exists, so
  you can exercise the loop before the pipeline is built. This is the fast
  build-and-debug tool; `app.py` is the eventual user-facing UI.
- **`app.py`** — user-facing entry point. Holds the chat loop + session memory.
  Depends on `rag/`, knows nothing about `ingest/`.
- **`sources.txt`** — plain list of seed URLs, read by `crawl.py`.
- **`chroma_db/`** — persisted vector store holding the **child** embeddings.
  Written by `ingest/`, read by `rag/`. Gitignore it.
- **`docstore/`** — persisted **parent** chunks for Small2Big. `ParentDocumentRetriever`
  matches on child embeddings in Chroma, then looks up the parent here to send to
  the LLM. This is a *separate* store from Chroma — parents do **not** live in the
  vector DB. Without it, parents fall back to in-memory and vanish on restart.
  Gitignore it.
- **`config.py`** — central settings file. Both pipelines import from it, so you
  switch models/embeddings/tuning in one place without touching pipeline code.
- **`requirements.txt` / `.env` / `.gitignore` / `README.md`** — project scaffolding.
  `.env` holds secrets and host settings the config reads (never committed);
  `.gitignore` must list `chroma_db/`, `docstore/`, `.env`, and `__pycache__`;
  `README.md` states the run order (ingest first, then `app.py`).

Dependency direction: nothing in `rag/` ever imports from `ingest/`. The two
stores — `chroma_db/` (children) and `docstore/` (parents) — are the contract
between them. `eval/` imports `rag/` only.

### Configuration (`config.py`)

All switchable settings live in `config.py`. Each value also reads from an
environment variable, so you can override without editing the file.

| Variable | Purpose | Default |
|----------|---------|---------|
| `ORGANIZATION_NAME` | University name injected into the prompt template | `Your University` |
| `LLM_MODEL` | Generation model (any Ollama model you've pulled) | `qwen2.5:7b-instruct` |
| `EMBEDDING_MODEL` | Embedding model (Ollama) | `nomic-embed-text` |
| `OLLAMA_BASE_URL` | Where Ollama is listening | `http://localhost:11434` |
| `LLM_TEMPERATURE` | Sampling temperature (0.0 = factual) | `0.0` |
| `CHROMA_PERSIST_DIR` | On-disk child-vector store location | `./chroma_db` |
| `DOCSTORE_DIR` | On-disk parent-chunk store for Small2Big | `./docstore` |
| `CHROMA_COLLECTION` | Collection name in Chroma | `university_pages` |
| `CHILD_CHUNK_SIZE` / `PARENT_CHUNK_SIZE` | Small2Big chunking knobs (tokens) | `200` / `800` |
| `CHUNK_OVERLAP` | Overlap between child chunks (tokens) | `40` |
| `SOURCES_FILE` / `CRAWL_MAX_DEPTH` | Crawl seed list + depth | `sources.txt` / `2` |
| `RETRIEVAL_K` / `RERANK_TOP_N` | Chunks retrieved / kept after rerank | `20` / `5` |
| `NUM_QUERY_VARIANTS` | Variants generated for multi-query expansion | `4` |
| `USE_HYBRID_SEARCH` | Vector + BM25 ensemble | `true` |
| `USE_RERANKER` | Cross-encoder rerank | `true` |
| `USE_MULTI_QUERY` | RAG Fusion expansion | `true` |
| `USE_CRAG` | Pre-generation relevance check + retry (needs a 7b+ judge; false-negatives on 0.5b) | `false` |
| `USE_REVERSE_HYDE` | Match against generated questions | `true` |
| `LANGSMITH_TRACING` / `LANGSMITH_PROJECT` | Observability | `false` / `university-rag` |

Every Modular RAG feature is a boolean toggle. That is the point of modular RAG:
flip a flag, re-run RAGAS, keep what helps. Note the hybrid/rerank/multi-query/
CRAG flags default to **true** — they earn their place for a university Q&A
workload.

> ⚠️ **Changing `EMBEDDING_MODEL`, the chunk sizes, or `USE_REVERSE_HYDE`
> requires re-running the ingestion pipeline** (`build_index.py`), which rebuilds
> **both** stores (`chroma_db/` and `docstore/`). Vectors and the stored
> child/parent/question artifacts are only consistent when produced by the same
> settings used at query time — mixing them silently breaks retrieval.

---

## 6. Dependencies

These go in `requirements.txt` (pin versions once your first build works):

```
langchain  langchain-community  langchain-ollama
langchain-chroma  chromadb  trafilatura
rank_bm25                 # hybrid search (now default)
sentence-transformers     # reranking (now default)
ragas  datasets           # evaluation harness
streamlit                 # if you want a UI
```

The parent docstore for Small2Big uses LangChain's built-in
`LocalFileStore` / storage helpers — no extra dependency needed.

---

## 7. Recommended Build Order

Build the linear core first, get it working end-to-end, then add modules one at a
time — measuring each with RAGAS before keeping it.

1. **Core path.** Ingestion on ~10 key pages → bare RAG chain in a notebook →
   eyeball answers + citations, LangSmith on. (Fixed chunking is fine here.)
2. **Terminal CLI.** Wire `cli.py` to the pipeline as your build-and-debug loop —
   faster to iterate against than a notebook or a browser, and it exercises the
   same `rag/` interface `app.py` will later use.
3. **Evaluation harness.** Write ~30 ground-truth Q/A pairs across the kinds of
   question your bot will field; wire up `eval/run_ragas.py`. *Do this early —
   every later step is measured against it.*
4. **Hybrid search on by default.** Add BM25 to the ensemble; re-run RAGAS.
5. **Multi-query expansion (RAG Fusion)** + RRF merge; re-run RAGAS.
6. **Reranker** non-optional; tune `RERANK_TOP_N`; re-run RAGAS.
7. **Small2Big chunking.** Switch to parent/child; re-ingest; re-run RAGAS.
8. **Reverse HyDE + enriched metadata** at ingestion; re-ingest; re-run RAGAS.
9. **CRAG relevance check** + retry branch; re-run RAGAS.
10. **History-aware retrieval** for multi-turn.
11. **Wrap in Streamlit** (`app.py`) once you want a browser UI for end users.

If a step doesn't move the RAGAS numbers, toggle it back off. Modularity means you
can.

---

## 8. Evaluation (RAGAS)

You cannot tune what you cannot measure — this is the biggest structural gap a
v1 plan leaves open. Build a fixed test set and score the pipeline after every
change.

| Metric | What it measures | Guards |
|--------|------------------|--------|
| Context Relevance | Are retrieved chunks actually relevant? | Retrieval quality |
| Answer Faithfulness | Does the answer stay grounded in the chunks? | Hallucination |
| Answer Relevance | Does the answer address the question? | Generation quality |
| Noise Robustness | Does irrelevant retrieval derail the answer? | End-to-end robustness |

Test set: ~30 question/answer pairs, covering the main kinds of question your
bot will field and a few deliberately unanswerable ones to test the "I don't
know" guardrail. Store as `eval/testset.jsonl`; log runs to LangSmith so you can
diff scores across changes.

---

## 9. Practical Tips

- **Start small**: 10–20 key pages (admissions, courses, fees, contacts).
- **Hybrid search is default now** — course codes, building names, and acronyms
  are exact-match problems vector search handles poorly.
- **Measure before you keep** — every module is a toggle; let RAGAS decide.
- **Citations build trust** — always show source URLs.
- **Scope the crawler** tightly or you will index junk.
- **Respect `robots.txt`** and only crawl pages you are allowed to.
- **Watch the latency budget** — multi-query × hybrid × CRAG retries multiply
  calls. On a laptop, cap `NUM_QUERY_VARIANTS` and run variants concurrently.
- **Turn on LangSmith from day one** — fastest way to debug bad answers
  (usually retrieval, not the LLM).

---

## 10. Where this sits

This design is squarely in **Modular RAG** territory: expansion, fusion, and an
adaptive self-check (CRAG) replace the fixed retrieve-then-read chain, and every
component is independently swappable from `config.py`. The remaining frontier
— iterative/adaptive multi-hop retrieval, fine-tuned retrievers, RAG+fine-tuning
hybrids — is real but overkill for a single-domain university Q&A bot. Stop here
and tune; you'll get more from a good RAGAS loop than from adding more machinery.
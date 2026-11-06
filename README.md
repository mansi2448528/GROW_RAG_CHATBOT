# HDFC Mutual Fund FAQ Assistant

A facts-only RAG prototype over **one AMC page and four HDFC scheme pages** on
Groww. Every answer is a single factual claim with exactly one official source and
a freshness date. Advice, suitability, performance, PII, and out-of-scope questions
are refused.

The demo is deliberately inspectable: retrieval and generation write JSON artifacts,
and the UI's **How this answer was produced** panel renders those files directly.

> **Facts-only. No investment advice.** Answers are drawn from official public pages
> and are not a recommendation to buy or sell any scheme. Mutual fund investments
> are subject to market risks; read all scheme-related documents carefully.

---

## Scope

| Covered | Not covered |
|---|---|
| HDFC Large Cap Fund (Direct-Growth) | Other AMCs |
| HDFC Flexi Cap Fund (Direct-Growth) | Advice, opinions, rankings |
| HDFC ELSS Tax Saver (Direct-Growth) | Returns, CAGR, future performance |
| HDFC Mid Cap Fund (Direct-Growth) | Portfolio/suitability questions |
| HDFC Mutual Fund AMC overview | PAN, folio, account, email, phone data |

The corpus is the five URLs in `data/corpus/urls.csv`. `docs/SOURCES.md` lists them
with the reason each was included.

---

## Quick start (Windows PowerShell)

```powershell
cd "D:\NEXTLEAP PM\mf-faq"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Build the index (only needed once, or after corpus/config changes)
python -m app.pipeline --stage all

# Start the UI
streamlit run app/ui.py
```

Open the URL Streamlit prints (usually <http://localhost:8501>).

**No `.env` or Groq key is required.** Answers are always composed with
deterministic extractive templates, so retrieval, the UI, and the tests run
offline. The generation artifact records `llm_used=false` instead of pretending an
LLM was called.

**Optional rephrasing layer.** Copy `.env.example` to `.env` and set
`GROQ_API_KEY` plus `LLM_MODEL` (for example `openai/gpt-oss-120b`) to let a Groq
model rewrite the already-verified sentence. The model is never allowed to supply
numbers, names or citations: its output must pass the same validators plus an
exact number/name-preservation check, otherwise the extractive answer is kept and
the fallback reason is recorded in `06_generation/<qid>.json`. A missing key, an
API error, or malformed JSON degrades to the extractive path without breaking the
UI.

On this machine the interpreter is not on `PATH`, so the equivalent commands are:

```powershell
& .\.venv\Scripts\python.exe -m app.pipeline --stage all
& .\.venv\Scripts\python.exe -m streamlit run app/ui.py
```

---

## Three-minute demo

1. **Onboarding shell** — open the app. The collapsible left sidebar holds
   **New Chat**, chat search, session history, **Settings**, and a guest profile.
   The welcome panel is led by the large **HDFC Mutual Fund AI Assistant**
   heading, with `How can I help you today?` as its tagline and four suggested
   questions above the fixed input box.
2. **Answer path** — click **“Expense ratio of HDFC Flexi Cap Direct?”**. The reply
   shows the claim, a clickable Groww source, and `Last updated from sources`; the
   Copy and Regenerate controls sit under the answer.
3. **Chat history** — ask a second question, then click **New Chat**. The previous
   conversation stays in the sidebar and reopens with its answers intact; the search
   box filters the list.
4. **Refusal path** — ask `Should I buy HDFC Mid Cap?`. The safety gate refuses
   without searching, and the reply carries no scheme citation. This is intentional:
   the refusal must not read as a soft recommendation.
5. **No-context path** — ask `Exit load after 5 years?`. The corpus only discloses
   the “within 1 year” condition, so the reply says it could not find the answer
   rather than reusing a different row.
6. **Filter hint** — choose a scheme under **Settings → Optional filter hint**, then
   ask `What is the exit load?`. The selector narrows retrieval only when the
   question does not name a scheme; a scheme named in the question always wins.
7. **Trace panel** — open **How this answer was produced**. The five tabs read
   `05_retrieval/<query_id>.json` and `06_generation/<query_id>.json` directly:
   Gate · Retrieval · Prompt · Raw output · Validators. With a key configured, the
   Prompt and Raw output tabs show the actual rephrasing prompt and model response;
   without one, they state honestly that no LLM was called. The panel can be hidden
   with **Settings → Show answer trace**.

A PII question such as `My PAN is ABCDE1234F, check my folio` refuses before
retrieval and writes **no** retrieval trace, **no** generation trace, and **no**
`qa_log` row. The chat displays `[personal data withheld]` instead of the question,
titles the conversation `Personal data question`, and explains the missing trace.

---

## How the answer is produced

```
question
  └─ services/query.py
       ├─ pre-gate (PII / advice / performance / scope)  → refusal card
       ├─ src/retriever.py  gate → analyse → filter → search → dedupe → threshold
       │                    writes 05_retrieval/<qid>.json
       ├─ src/generator.py  extractive template → validators → answer/no_context
       │                    └─ optional src/llm.py rephrase → validators
       │                    writes 06_generation/<qid>.json
       └─ AnswerCard        claim · source_url · last_updated
```

Answers are never written by the UI. The Streamlit page only calls
`services.query.QueryService.answer()` and renders the returned card plus the two
artifact files.

**Validators** block an answer if the citation is not in the retrieved hits and the
corpus manifest, the answer exceeds three sentences, advice/performance language
appears, PII is echoed, a URL appears inside the answer, or the asked value is
missing. A failed extractive answer is downgraded to `no_context`. A failed LLM
rewrite is not downgraded: the extractive answer is kept, because it already passed
those checks.

---

## Artifacts

Each stage writes its own folder under `data/artifacts/`:

| Stage | Artifact |
|---|---|
| Load | `01_loaded/<doc_id>.jsonl`, `load_report.csv` |
| Chunk | `02_chunks/chunks.jsonl`, `chunks_preview.md`, `chunk_stats.csv` |
| Embed | `03_embeddings/embeddings.npy`, `embedding_manifest.json` |
| Store | `04_vectordb/collection_report.md` |
| Retrieve | `05_retrieval/<qid>.json`, `eval_report.md`, `retrieval_eval.json` |
| Generate | `06_generation/<qid>.json` |
| Audit | `qa_log.csv` (non-PII queries only) |

Retrieval evaluation is reproducibly **10/10 hit@3 at rank 1** on the probe set:

```powershell
python -m app.pipeline --eval-retrieval
```

---

## Tests

```powershell
pytest -q
```

Expected: **81 passed**. The suite covers parsing, chunking, embeddings, the Chroma
store, retrieval, generation/validators, the optional LLM rephrasing path, and the
UI query path. No test requires a Groq key or network access; the LLM tests inject a
fake phraser.

For the rendered page itself, run the AppTest smoke test:

```powershell
python eval/ui_smoke.py
```

It drives the real Streamlit page without a browser: sidebar shell, brand heading and
exact welcome tagline, exact input placeholder, the four suggestions,
answer/refusal/no-context cards, copy and regenerate controls, chat history/search,
the scheme selector, the trace tabs, the trace-visibility setting, and the PII path. It writes real query
artifacts and `qa_log.csv` rows (that is the point of a smoke test). Streamlit may
print a harmless `missing ScriptRunContext` warning; the checks still run.

---

## Deploy

See **[DEPLOY.md](DEPLOY.md)** for step-by-step instructions.

- **Vercel:** deploy through the root `Dockerfile` using Vercel's Container
  Runtime (`framework: container`). Set `PORT=8501` and, optionally,
  `GROQ_API_KEY` / `LLM_MODEL`. See the Vercel section in the deployment guide
  for the beta WebSocket session limit.
- **Streamlit Community Cloud:** free, native Streamlit, supports dashboard
  secrets.
- **Any Docker host:** use the included `Dockerfile` (Render, Railway, Fly.io,
  Azure, etc.).

The prebuilt `data/chroma/` index and `data/corpus/urls.csv` manifest are
committed because the deployed app does not run the ingestion pipeline.

---

## Layout

```
mf-faq/
├─ app/
│  ├─ pipeline.py        # CLI: load → chunk → embed → store, query/eval
│  └─ ui.py              # Streamlit page (presentation only)
├─ config/
│  ├─ config.yaml        # models, sizes, thresholds, paths
│  ├─ lexicon.yaml       # scheme aliases, fact keys, refusal patterns
│  └─ config.py          # typed loader + validation
├─ services/query.py     # single request path used by the UI
├─ src/                  # loader, chunker, embedder, store, retriever, generator, llm
├─ tests/                # 81 tests
├─ data/                 # corpus, cache, Chroma index, artifacts
└─ docs/SOURCES.md
```

---

## Known limits

- **Five-page corpus.** The brief asks for 15–25 URLs, but only the five supplied
  URLs are ingested. `docs/SOURCES.md` records this honestly; nothing was padded.
- **No education link.** The corpus has no educational page, so a refusal cannot
  show one. The UI says so rather than inventing a scheme citation.
- **No performance answers.** Return/performance data is filtered out by design.
- **Live pages change.** `Last updated from sources` comes from the snapshot date.
  Re-run `python -m app.pipeline --stage load --no-cache` to refresh the snapshot.
- **Trace tables use HTML.** `st.dataframe` needs `pyarrow`/NumPy 2, while this
  project pins `numpy==1.26.4`; the trace panel renders small HTML tables instead.
- **“Exit load after 5 years”** returns `no_context` because the page only states
  the “within 1 year” condition.
- **LLM wording varies.** With a key configured, phrasing may differ between runs;
  numbers and citations do not, because they come from the extractive row and a
  rewrite that changes them is rejected. Leave `LLM_MODEL` blank for fully
  deterministic output.

Implementation details live in `../implementation.md` and `../architecture.md`.

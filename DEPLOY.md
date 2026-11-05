# Deployment guide

This project is a **Streamlit** application backed by a local Chroma index and a
local sentence-transformers model. It needs a long-lived Python process with a
writable filesystem and WebSocket support.

## Important: Vercel will not run this app

Vercel is built for static files and short-lived serverless functions. It does
not run a persistent Streamlit server, does not support Streamlit's WebSocket
session protocol, and its serverless bundle/time limits are far below what
`chromadb` + `sentence-transformers` + `torch` require. Pushing this repository
to Vercel will build, but the app will not work.

Use one of the options below instead. If Vercel is a hard requirement, the UI
and the RAG backend must be re-architected as separate services, with the
backend hosted somewhere that can run Python containers.

## Option 1 — Streamlit Community Cloud (recommended, free)

1. Push this repository to GitHub (branch `main`).
2. Sign in at <https://share.streamlit.io> with the GitHub account that owns the
   repository.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Configure:
   - Repository: `mansi2448528/GROW_RAG_CHATBOT`
   - Branch: `main`
   - Main file path: `app/ui.py`
   - Python version: **3.12** (Advanced settings)
5. In **Advanced settings → Secrets**, paste:

   ```toml
   GROQ_API_KEY = "your-key-here"
   LLM_MODEL = "openai/gpt-oss-120b"
   ```

   The app also runs without these values; answers then use the verified
   extractive wording only.
6. Click **Deploy**.

The repository already contains:

- `requirements.txt` with the CPU-only PyTorch index
- `packages.txt` (`libgomp1`)
- `.streamlit/config.toml` (dark theme, usage stats off)
- `data/chroma/` — the prebuilt retrieval index required at startup
- `data/corpus/urls.csv` and `config/` — corpus manifest, lexicon and settings

The embedding model is downloaded from Hugging Face on first start and cached
for later runs, so the first load can take a few minutes.

## Option 2 — Docker host (Render, Railway, Fly.io, Azure, etc.)

The included `Dockerfile` runs the app as a non-root user on port `8501`.

```powershell
docker build -t hdfc-mf-faq .
docker run --rm -p 8501:8501 `
  -e GROQ_API_KEY="your-key" `
  -e LLM_MODEL="openai/gpt-oss-120b" `
  hdfc-mf-faq
```

Open <http://localhost:8501>. Configure the same two environment variables on
the hosting provider. Do **not** commit a real `.env` file.

## Secrets

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | No | Enables optional AI-assisted rephrasing |
| `LLM_MODEL` | No | Groq model id used for rephrasing |

Without both values the app is fully functional and deterministic. On Streamlit
Community Cloud, values entered under **Secrets** are copied into the process
environment at startup by `app/ui.py`; locally, `.env` is read by
`config/config.py`.

## Health check

Streamlit exposes `/_stcore/health`. The Dockerfile uses it as its health check.

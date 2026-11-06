# Deployment guide

This is a **Streamlit** application backed by a prebuilt local Chroma index and a
local sentence-transformers model. It needs a Python process that can serve
HTTP **and** WebSocket traffic, plus a writable directory for runtime state.

The repository ships two deployment paths:

1. **Vercel** using the Container Runtime (`Dockerfile` at the project root).
2. **Streamlit Community Cloud** or any Docker host (Render, Railway, Fly.io,
   Azure, etc.).

The prebuilt `data/chroma/` index and `data/corpus/urls.csv` manifest are
committed because the deployed app does not run the ingestion pipeline.

---

## Option 1 — Vercel (Container Runtime)

Vercel Functions can build and run the root `Dockerfile` as an OCI container
image. The container serves Streamlit on the port given by `PORT`.

> Container Runtime and WebSocket support are **public beta** on all Vercel
> plans. On the Hobby plan a single WebSocket connection is capped at about
> **300 seconds**, so a long chat session reconnects periodically. The page
> reconnects automatically; history held only in `st.session_state` may reset.

### 1. Prerequisites

- A Vercel account.
- The Vercel CLI (optional but used below):

  ```powershell
  npm.cmd install -g vercel
  vercel --version
  ```

### 2. Authenticate

```powershell
vercel login
# or, for CI / non-interactive use:
vercel --token YOUR_VERCEL_TOKEN
```

### 3. Link the project

From the repository root:

```powershell
vercel link
```

When asked, create a new project and keep the repository root as the project
root. If the project already exists, set the framework explicitly:

```powershell
vercel project update --framework container --yes
```

The repository already contains `vercel.json` (`"framework": "container"`), so a
plain `vercel` deploy also selects the container preset.

### 4. Configure environment variables

Set these in the Vercel project (Settings → Environment Variables), for
Production and Preview:

| Variable | Required | Value |
|---|---|---|
| `PORT` | **Yes** | `8501` |
| `GROQ_API_KEY` | No | Enables optional AI-assisted rephrasing |
| `LLM_MODEL` | No | Groq model id, e.g. `openai/gpt-oss-120b` |

`PORT=8501` must match the port in the image. Without it Vercel routes to port
80; the image can bind either port, but the setting removes the ambiguity.

Do **not** add a real `.env` file. `.env` is excluded by `.gitignore`,
`.dockerignore` and `.vercelignore`.

### 5. Deploy

```powershell
vercel --prod
```

The first build installs the CPU-only PyTorch stack and pre-downloads the
embedding model, so it takes several minutes. Later builds reuse cached layers.

### 6. Verify

```powershell
vercel inspect
```

Open the production URL and check:

- `https://<your-app>.vercel.app/_stcore/health` returns `ok`.
- The welcome screen shows the **HDFC Mutual Fund AI Assistant** heading and the
  four suggestion chips.
- A sample question returns an answer with citations and the
  `Last updated from sources:` stamp.

### How the container adapts to Vercel

- The deployment filesystem is read-only. `deploy/vercel_entrypoint.sh` copies
  the Chroma index to `/tmp/mf-faq/chroma` and redirects retrieval/generation
  artifacts and the corpus cache to `/tmp`.
- Generated query artifacts are therefore **ephemeral** and reset when an
  instance is recycled.
- The embedding model is baked into the image at `/opt/hf` and copied into the
  writable Hugging Face cache at startup, so requests do not depend on a
  Hugging Face download.
- The app runs as the non-root user `app`. `libcap2-bin` grants permission to
  bind a low port if the platform assigns one.

### Troubleshooting

- **404 or connectivity errors on every route:** confirm `PORT=8501` is set for
  the environment you deployed and that the framework is `container`.
- **`/_stcore/health` never becomes ready:** check the runtime logs with
  `vercel logs <deployment-url>`. A missing `data/chroma` index or corpus
  manifest is the usual cause.
- **Session drops after ~5 minutes:** expected on the Hobby plan due to the
  WebSocket duration cap; use a Pro plan or another host for long sessions.

---

## Option 2 — Streamlit Community Cloud (free)

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

---

## Option 3 — Docker host (Render, Railway, Fly.io, Azure, etc.)

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

---

## Secrets

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | No | Enables optional AI-assisted rephrasing |
| `LLM_MODEL` | No | Groq model id used for rephrasing |

Without both values the app is fully functional and deterministic. On Streamlit
Community Cloud, values entered under **Secrets** are copied into the process
environment at startup by `app/ui.py`. On Vercel and Docker hosts, set them as
environment variables. Locally, `.env` is read by `config/config.py`.

## Health check

Streamlit exposes `/_stcore/health`, which both the Dockerfile health check and
Vercel use to confirm the app is serving.

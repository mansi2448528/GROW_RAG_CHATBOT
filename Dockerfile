FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MF_FAQ_DATA_ROOT=/tmp/mf-faq \
    PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# libgomp is required by the CPU PyTorch wheel. libcap2-bin lets the non-root
# user bind the platform-assigned port even when that port is below 1024.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libcap2-bin \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Pre-download the embedding model at build time so cold starts do not depend
# on Hugging Face being reachable, and the first query is not a long download.
COPY config/config.yaml ./config/config.yaml
RUN HF_HOME=/opt/hf TRANSFORMERS_CACHE=/opt/hf python -c \
    "import yaml; from sentence_transformers import SentenceTransformer; \
cfg = yaml.safe_load(open('config/config.yaml', encoding='utf-8')); \
SentenceTransformer(cfg['embedding']['model_id'])" \
    && chmod -R a+rX /opt/hf

COPY deploy/vercel_entrypoint.sh /usr/local/bin/vercel-entrypoint.sh
RUN chmod +x /usr/local/bin/vercel-entrypoint.sh

COPY . .

RUN chown -R app:app /app \
    && setcap 'cap_net_bind_service=+ep' "$(readlink -f "$(command -v python3)")"
USER app

# Vercel routes to $PORT (project setting). EXPOSE is documentation for
# generic Docker hosts.
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/_stcore/health' % os.environ.get('PORT', '8501'), timeout=5)" || exit 1

CMD ["/usr/local/bin/vercel-entrypoint.sh"]

#!/bin/sh
# Vercel container Functions mount the deployment read-only. Streamlit and
# Chroma both need a writable directory, so copy the prebuilt index into /tmp
# once per container instance and redirect all generated artifacts there.
set -eu

DATA_ROOT="${MF_FAQ_DATA_ROOT:-/tmp/mf-faq}"
HF_CACHE="${HF_HOME:-/tmp/hf}"
SOURCE_CHROMA="${MF_FAQ_SOURCE_CHROMA:-/app/data/chroma}"

mkdir -p "$DATA_ROOT" "$HF_CACHE"

if [ ! -d "$DATA_ROOT/chroma" ]; then
    cp -a "$SOURCE_CHROMA" "$DATA_ROOT/chroma"
fi
mkdir -p "$DATA_ROOT/artifacts" "$DATA_ROOT/cache"

# The image bakes the embedding model into /opt/hf. Copy it into the writable
# cache so sentence-transformers can create its lock files at runtime.
if [ -d /opt/hf/hub ] && [ ! -d "$HF_CACHE/hub" ]; then
    cp -a /opt/hf/. "$HF_CACHE"/
fi

export MF_FAQ_CHROMA_DIR="$DATA_ROOT/chroma"
export MF_FAQ_ARTIFACTS_DIR="$DATA_ROOT/artifacts"
export MF_FAQ_CACHE_DIR="$DATA_ROOT/cache"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

exec streamlit run app/ui.py \
    --server.address=0.0.0.0 \
    --server.port="${PORT:-8501}" \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false

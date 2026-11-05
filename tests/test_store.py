"""Vector-store invariants (implementation.md §5). No network, no model weights."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")

from config.config import load_config
from src.artifacts import StageRecorder
from src.embedder import embed_all
from src.loader import PipelineError
from src.models import Chunk, DocMeta
from src.store import (
    EmbeddingMismatchError,
    Store,
    StoreError,
    collection_exists,
    open_collection,
    store_all,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


class FakeEmbedder:
    """Deterministic unit vectors; counts as the encoder for the whole test."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts):
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            vector = np.random.default_rng(seed).random(self.dim).astype(np.float32)
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.vstack(rows).astype(np.float32)


def make_chunk(ordinal: int, *, doc_id: str = "test_doc", text: str | None = None) -> Chunk:
    body = text or f"Exit load: Nil ({ordinal})"
    meta = DocMeta(doc_id=doc_id, source_url=f"https://groww.in/{doc_id}",
                   page_title="HDFC Test Fund", page_type="groww_scheme",
                   scheme="hdfc_elss", amc="HDFC Mutual Fund", plan="direct_growth",
                   fetched_at=date(2026, 11, 1))
    return Chunk(
        chunk_id=f"{doc_id}#{ordinal:02d}", doc_id=doc_id, text=body,
        embed_text=f"HDFC Test Fund | Fund facts | {body}",
        section_path=["Fund facts"], kind="definition", fact_keys=["exit_load"],
        is_atomic=True, is_quarantined=False, token_count=8, meta=meta,
        content_sha256="abc123", run_id="chunk-run",
    )


def isolated_cfg(cfg, tmp_path):
    clone = cfg.model_copy(deep=True)
    clone.app.artifacts_dir = tmp_path / "artifacts"
    clone.vectorstore.persist_dir = tmp_path / "chroma"
    for cached in ("fingerprint", "collection_name", "config_hash"):
        clone.__dict__.pop(cached, None)
    return clone


def seed_embeddings(cfg, chunks):
    """Write the chunk + embedding artifacts a real run would leave on disk."""
    StageRecorder(cfg, run_id="chunk-run").jsonl(
        "02_chunks", "chunks.jsonl", [c.to_dict() for c in chunks])
    return embed_all(cfg, StageRecorder(cfg, run_id="embed-run"),
                     chunks=chunks, embedder=FakeEmbedder())


def test_fingerprint_mismatch_raises(cfg):
    client = chromadb.EphemeralClient()
    store = Store(cfg, client=client)
    try:
        store.ensure(name="probe_fp", fingerprint="aaaa1111", dim=384)
        with pytest.raises(EmbeddingMismatchError, match="--rebuild"):
            store.ensure(name="probe_fp", fingerprint="bbbb2222", dim=384)
    finally:
        client.delete_collection("probe_fp")


def test_rebuild_replaces_a_stale_collection(cfg):
    client = chromadb.EphemeralClient()
    store = Store(cfg, client=client)
    first = store.ensure(name="probe_rebuild", fingerprint="aaaa1111", dim=3)
    first.upsert(ids=["x"], embeddings=[[1.0, 0.0, 0.0]], documents=["old"],
                 metadatas=[{"scheme": "hdfc_elss"}])

    rebuilt = store.ensure(name="probe_rebuild", fingerprint="bbbb2222", dim=3,
                           rebuild=True)

    assert rebuilt.metadata["fingerprint"] == "bbbb2222"
    assert rebuilt.count() == 0
    client.delete_collection("probe_rebuild")


def test_store_roundtrip_writes_loadable_artifacts(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    chunks = [make_chunk(i) for i in range(4)]
    seed_embeddings(small, chunks)

    recorder = StageRecorder(small, run_id="store-run")
    result = store_all(small, recorder)

    assert result.status.count == 4
    assert result.status.per_doc == {"test_doc": 4}
    assert result.status.min_cosine > 1 - 1e-4
    assert result.status.nearest_is_self
    assert result.status.fingerprint == small.fingerprint

    stage = small.stage_dir("04_vectordb")
    report = (stage / "collection_report.md").read_text("utf-8")
    assert "- [x] chroma count equals chunk count (4 = 4)" in report
    assert "metadata is scalar" in report
    assert "`test_doc#00`" in report

    manifest = json.loads((stage / "collection_manifest.json").read_text("utf-8"))
    assert manifest["count"] == manifest["chunk_count"] == 4
    assert manifest["collection"] == small.collection_name
    assert manifest["fingerprint"] == small.fingerprint
    assert manifest["dim"] == 384 and manifest["distance"] == "cosine"

    # Persistence: a fresh client sees the same collection and metadata.
    reopened = Store(small)
    collection = reopened.client.get_collection(small.collection_name)
    assert collection.metadata["fingerprint"] == small.fingerprint
    assert collection.count() == 4
    assert collection_exists(small)
    assert open_collection(small).count() == 4


def test_rerun_is_idempotent_and_deletes_stale_ids(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    chunks = [make_chunk(i) for i in range(4)]
    seed_embeddings(small, chunks)
    store_all(small, StageRecorder(small, run_id="r1"))

    # The corpus shrank to 3 chunks: the store must not keep the fourth.
    kept = chunks[:3]
    seed_embeddings(small, kept)
    again = store_all(small, StageRecorder(small, run_id="r2"))

    assert again.status.count == 3
    assert again.status.stale_deleted == 1
    collection = Store(small).client.get_collection(small.collection_name)
    assert sorted(collection.get(include=[])["ids"]) == [c.chunk_id for c in kept]


def test_store_refuses_embeddings_for_other_chunks(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    seed_embeddings(small, [make_chunk(0), make_chunk(1)])

    # chunks.jsonl is replaced with a different set of the same size, as a stale
    # chunk stage would leave it: the ids digest must catch it before any write.
    different = [make_chunk(0), make_chunk(2)]
    StageRecorder(small, run_id="chunk-run-2").jsonl(
        "02_chunks", "chunks.jsonl", [c.to_dict() for c in different])

    with pytest.raises(StoreError, match="ids_sha256"):
        store_all(small, StageRecorder(small, run_id="r3"))


def test_store_requires_embeddings(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    stage = small.stage_dir("02_chunks")
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "chunks.jsonl").write_text(
        json.dumps(make_chunk(0).to_dict(), default=str) + "\n", encoding="utf-8")

    with pytest.raises(PipelineError, match="embed"):
        store_all(small, StageRecorder(small, run_id="r4"))


def test_open_collection_explains_missing_index(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)

    with pytest.raises(StoreError, match="--stage store"):
        open_collection(small)


def test_open_collection_refuses_fingerprint_drift(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    seed_embeddings(small, [make_chunk(0), make_chunk(1)])
    store_all(small, StageRecorder(small, run_id="r5"))

    # Model changes normally produce a *new* collection name (name embeds the
    # fingerprint). The alarm also has to fire if the metadata was tampered with
    # or written by another tool.
    collection = Store(small).client.get_collection(small.collection_name)
    tampered = {k: v for k, v in collection.metadata.items() if k != "hnsw:space"}
    tampered["fingerprint"] = "deadbeef0000"
    collection.modify(metadata=tampered)

    with pytest.raises(EmbeddingMismatchError, match="--stage store --rebuild"):
        open_collection(small)

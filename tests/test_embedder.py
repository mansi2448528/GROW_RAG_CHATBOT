"""Embedding invariants (implementation.md §4). Network-free by default.

A fake encoder exercises the artefact/validation logic; one optional test uses the
real MiniLM weights when they are available, because "384 numbers came back" is
not evidence that the vectors mean anything.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import date

import numpy as np
import pytest

from config.config import PROJECT_ROOT, load_config
from src.artifacts import StageRecorder
from src.embedder import (
    Embedder,
    EmbeddingError,
    EmbeddingRun,
    embed_all,
    embedder_check,
    ids_digest,
    load_chunks,
    validate_vectors,
)
from src.loader import PipelineError
from src.models import Chunk, DocMeta

requires_model = pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed",
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make_chunk(ordinal: int, *, doc_id: str = "test_doc", fact_key: str = "exit_load",
               text: str | None = None) -> Chunk:
    body = text or f"Exit load: Nil ({ordinal})"
    meta = DocMeta(doc_id=doc_id, source_url="https://groww.in/test",
                   page_title="HDFC Test Fund", page_type="groww_scheme",
                   scheme="hdfc_elss", amc="HDFC Mutual Fund", plan="direct_growth",
                   fetched_at=date(2026, 11, 1))
    return Chunk(
        chunk_id=f"{doc_id}#{ordinal:02d}", doc_id=doc_id, text=body,
        embed_text=f"HDFC Test Fund | Fund facts | {body}",
        section_path=["Fund facts"], kind="definition", fact_keys=[fact_key],
        is_atomic=True, is_quarantined=False, token_count=8, meta=meta,
        content_sha256="abc123", run_id="chunk-run",
    )


class FakeEmbedder:
    """Deterministic stand-in: same text -> same unit vector, no torch."""

    def __init__(self, dim: int = 384, *, normalize: bool = True,
                 zero_rows: set[int] | None = None) -> None:
        self.dim = dim
        self.normalize = normalize
        self.zero_rows = zero_rows or set()
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        rows = []
        for i, text in enumerate(texts):
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            vector = np.random.default_rng(seed).random(self.dim).astype(np.float32)
            if i in self.zero_rows:
                vector[:] = 0
            if self.normalize:
                norm = np.linalg.norm(vector)
                if norm:
                    vector /= norm
            rows.append(vector)
        return np.vstack(rows).astype(np.float32)


def isolated_cfg(cfg, tmp_path):
    clone = cfg.model_copy(deep=True)
    clone.app.artifacts_dir = tmp_path
    # cached_property values (fingerprint, collection_name, config_hash) live in
    # __dict__; a deep copy would carry them across a model change.
    for cached in ("fingerprint", "collection_name", "config_hash"):
        clone.__dict__.pop(cached, None)
    return clone


def test_unnormalised_vectors_are_rejected(cfg):
    chunks = [make_chunk(0), make_chunk(1)]
    vectors = FakeEmbedder(normalize=False).encode([c.embed_text for c in chunks])

    with pytest.raises(EmbeddingError, match="not unit length"):
        validate_vectors(vectors, chunks, cfg)


def test_dim_mismatch_is_rejected(cfg):
    chunks = [make_chunk(0), make_chunk(1)]
    vectors = FakeEmbedder(dim=128).encode([c.embed_text for c in chunks])

    with pytest.raises(EmbeddingError, match="embedding.dim"):
        validate_vectors(vectors, chunks, cfg)


def test_zero_vector_is_rejected(cfg):
    chunks = [make_chunk(0), make_chunk(1)]
    vectors = FakeEmbedder().encode([c.embed_text for c in chunks])
    vectors[1] = 0

    with pytest.raises(EmbeddingError, match="zero vector"):
        validate_vectors(vectors, chunks, cfg)


def test_row_count_must_match_chunks(cfg):
    chunks = [make_chunk(0), make_chunk(1)]
    vectors = FakeEmbedder().encode([c.embed_text for c in chunks])[:1]

    with pytest.raises(EmbeddingError, match="vectors for"):
        validate_vectors(vectors, chunks, cfg)


def test_ids_digest_is_order_sensitive():
    a, b = make_chunk(0), make_chunk(1)

    assert ids_digest([a, b]) == ids_digest([a, b])
    assert ids_digest([a, b]) != ids_digest([b, a])


def test_embed_stage_writes_loadable_artifacts(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    chunks = [make_chunk(i) for i in range(4)]
    fake = FakeEmbedder()

    recorder = StageRecorder(small, run_id="test-run")
    result = embed_all(small, recorder, chunks=chunks, embedder=fake)

    assert isinstance(result, EmbeddingRun)
    assert fake.calls[0] == [c.embed_text for c in chunks]      # encodes embed_text

    stage = tmp_path / "03_embeddings"
    matrix = np.load(stage / "embeddings.npy", allow_pickle=False)
    assert matrix.shape == (4, small.embedding.dim)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)

    manifest = json.loads((stage / "embedding_manifest.json").read_text("utf-8"))
    assert manifest["count"] == 4
    assert manifest["model_id"] == small.embedding.model_id
    assert manifest["fingerprint"] == small.fingerprint
    assert manifest["collection"] == small.collection_name
    assert manifest["ids_sha256"] == ids_digest(chunks)
    assert manifest["chunk_run_id"] == ""          # no 02_chunks/ manifest in tmp_path
    assert (stage / "sample_dims.md").exists()
    assert (stage / "stage_manifest.json").exists()
    # The manifest is the contract stage 5 reads; embedder_check re-validates it.
    assert embedder_check(small)["shape"] == (4, small.embedding.dim)


def test_manifest_detects_fingerprint_drift(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    chunks = [make_chunk(i) for i in range(2)]
    embed_all(small, StageRecorder(small, run_id="r1"), chunks=chunks, embedder=FakeEmbedder())

    drifted = small.model_copy(deep=True)
    drifted.embedding.model_id = "sentence-transformers/all-MiniLM-L12-v2"
    drifted.__dict__.pop("fingerprint", None)
    drifted.__dict__.pop("collection_name", None)

    with pytest.raises(EmbeddingError, match="fingerprint"):
        embedder_check(drifted)


def test_load_chunks_rejects_embed_text_drift(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    stage = small.stage_dir("02_chunks")
    stage.mkdir(parents=True, exist_ok=True)
    chunk = make_chunk(0)
    chunk.embed_text = "stale prefix | something else"
    (stage / "chunks.jsonl").write_text(
        json.dumps(chunk.to_dict(), default=str) + "\n", encoding="utf-8")

    with pytest.raises(EmbeddingError, match="embed_text"):
        load_chunks(small)


def test_load_chunks_requires_chunk_stage(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)

    with pytest.raises(PipelineError, match="chunk"):
        load_chunks(small)


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/artifacts/02_chunks/chunks.jsonl").exists(),
    reason="run --stage chunk first")
def test_real_chunks_are_embeddable(cfg):
    chunks = load_chunks(cfg)
    assert len(chunks) == len({c.chunk_id for c in chunks})
    assert len({c.doc_id for c in chunks}) == 5
    assert max(c.token_count for c in chunks) <= cfg.chunking.max_tokens
    assert all(c.text in c.embed_text for c in chunks)


@requires_model
@pytest.mark.skipif(
    not (PROJECT_ROOT / "data/artifacts/02_chunks/chunks.jsonl").exists(),
    reason="run --stage chunk first")
def test_real_model_embeds_the_corpus(cfg, tmp_path):
    chunks = load_chunks(cfg)
    small = isolated_cfg(cfg, tmp_path)
    runner = Embedder(small)
    try:
        runner.model          # downloads on first run; skip rather than fail offline
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"model unavailable here: {type(exc).__name__}")

    recorder = StageRecorder(small, run_id="real-run")
    result = embed_all(small, recorder, chunks=chunks, embedder=runner)

    assert result.vectors.shape == (len(chunks), 384)
    assert result.stats["norms_min"] > 0.999
    # The probes are the smoke test: real weights must land on a chunk that holds
    # the asked-for fact for at least 3 of the 4 queries.
    assert result.manifest["probes_passed"] >= 3, json.dumps(
        json.loads((small.stage_dir("03_embeddings") /
                    "embedding_manifest.json").read_text("utf-8")), indent=2)

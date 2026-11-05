"""Chunk → vectors, with the embedding bound to the index (implementation.md §4).

Three properties matter more than speed here:

1. **Encode `embed_text`, display `text`.** The embed text carries page title and
   section path, so a terse query ("exit load?") matches the section vocabulary
   rather than a bare table cell.
2. **Vectors are normalised float32**, checked after encoding — a silently
   un-normalised row makes cosine distance meaningless.
3. **The fingerprint is written down.** `embedding_manifest.json` records the
   model, dims and preprocessing version that produced `embeddings.npy`. Stage 5
   refuses to load a collection whose fingerprint differs, so a model swap can
   never silently search a different vector space.

Outputs (all inspectable without numpy except the `.npy` itself):
`03_embeddings/embeddings.npy`, `embedding_manifest.json`, `sample_dims.md`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from config.config import AppConfig, ConfigError
from src.artifacts import StageRecorder
from src.loader import PipelineError
from src.models import Chunk

#: Short queries used to prove the encoder is semantically sane before we trust
#: it with a citation. Not a retrieval test (§6 owns that) — a "did the weights
#: actually load" test. Expectations are (query, doc_id, fact_key).
PROBES: tuple[tuple[str, str, str], ...] = (
    ("What is the exit load for HDFC ELSS tax saver?", "groww_hdfc_elss", "exit_load"),
    ("ELSS lock-in period", "groww_hdfc_elss", "lock_in"),
    ("Expense ratio of HDFC Mid Cap Fund direct plan", "groww_hdfc_mid_cap", "expense_ratio"),
    ("Riskometer level of HDFC Flexi Cap Fund", "groww_hdfc_flexi_cap", "riskometer"),
)

SAMPLE_CHUNKS = 5
SAMPLE_DIMS = 8
NORM_TOLERANCE = 1e-3


class EmbeddingError(RuntimeError):
    """The vectors cannot be trusted: wrong shape, un-normalised, or a bad model."""


# ------------------------------------------------------------------ inputs

def chunks_path(cfg: AppConfig) -> Path:
    return cfg.stage_dir("02_chunks") / "chunks.jsonl"


def load_chunks(cfg: AppConfig) -> list[Chunk]:
    """Read the chunk stage's output, refusing anything the embedder can't trust."""
    path = chunks_path(cfg)
    if not path.exists():
        raise PipelineError(
            f"{path} is missing — run `python -m app.pipeline --stage chunk` first")
    lines = [line for line in path.read_text("utf-8").splitlines() if line.strip()]
    if not lines:
        raise PipelineError(f"{path} is empty — re-run `--stage chunk`")

    chunks = [Chunk.from_jsonl(line) for line in lines]

    duplicates = len(chunks) - len({c.chunk_id for c in chunks})
    if duplicates:
        raise EmbeddingError(f"{path}: {duplicates} duplicate chunk_id(s); embeddings.npy "
                             "is keyed by chunk_id, so duplicates would overwrite each other")
    over = [c for c in chunks if c.token_count > cfg.chunking.max_tokens]
    if over:
        raise EmbeddingError(
            f"{len(over)} chunk(s) exceed {cfg.chunking.max_tokens} tokens "
            f"(e.g. {over[0].chunk_id}) — re-run `--stage chunk`")
    if any(not c.text.strip() for c in chunks):
        raise EmbeddingError(f"{path}: empty chunk text; re-run `--stage chunk`")
    # Guards against a chunks.jsonl written by an older embed-text format: the
    # fingerprint covers preprocessing, but text/embed_text drift is cheap to catch.
    drifted = [c.chunk_id for c in chunks if c.text not in c.embed_text]
    if drifted:
        raise EmbeddingError(
            f"{len(drifted)} chunk(s) whose embed_text no longer contains their text "
            f"(e.g. {drifted[0]}) — re-run `--stage chunk`")
    return chunks


# ------------------------------------------------------------------ the model

class Embedder:
    """Thin wrapper so tests can inject a fake model and CI need no download."""

    def __init__(self, cfg: AppConfig, model=None) -> None:
        self.cfg = cfg
        self._model = model
        self.notes: list[str] = []

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:                       # noqa: BLE001
            raise ConfigError(
                "sentence-transformers is not installed — run "
                "`pip install -r requirements.txt`") from exc

        try:
            model = SentenceTransformer(self.cfg.embedding.model_id,
                                        device=self.cfg.embedding.device)
        except Exception as exc:                         # noqa: BLE001
            raise EmbeddingError(
                f"could not load {self.cfg.embedding.model_id}: {type(exc).__name__}: {exc}\n"
                "The first run downloads the model from Hugging Face; check network "
                "access or pre-download it, then re-run.") from exc

        model.max_seq_length = self.cfg.embedding.max_seq_length
        actual = int(model.get_sentence_embedding_dimension())
        if actual != self.cfg.embedding.dim:
            raise EmbeddingError(
                f"{self.cfg.embedding.model_id} emits {actual}-d vectors but "
                f"embedding.dim={self.cfg.embedding.dim} — fix config.yaml before "
                "building an index")
        return model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("nothing to encode")
        vectors = self.model.encode(
            list(texts),
            batch_size=self.cfg.embedding.batch_size,
            normalize_embeddings=self.cfg.embedding.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


# ------------------------------------------------------------------ checks

def validate_vectors(vectors: np.ndarray, chunks: Sequence[Chunk],
                     cfg: AppConfig) -> dict[str, float | int]:
    """Hard checks. Any failure means the index would be silently wrong."""
    if vectors.ndim != 2:
        raise EmbeddingError(f"expected a 2-D matrix, got shape {vectors.shape}")
    if vectors.shape[0] != len(chunks):
        raise EmbeddingError(f"{vectors.shape[0]} vectors for {len(chunks)} chunks")
    if vectors.shape[1] != cfg.embedding.dim:
        raise EmbeddingError(
            f"vectors are {vectors.shape[1]}-d but embedding.dim={cfg.embedding.dim}")
    if vectors.dtype != np.float32:
        raise EmbeddingError(f"expected float32, got {vectors.dtype}")

    if not np.isfinite(vectors).all():
        bad = int(np.argwhere(~np.isfinite(vectors))[0][0])
        raise EmbeddingError(f"non-finite values in vector {bad} ({chunks[bad].chunk_id})")

    norms = np.linalg.norm(vectors, axis=1)
    zeros = [chunks[i].chunk_id for i in np.flatnonzero(norms < 1e-8)]
    if zeros:
        raise EmbeddingError(f"zero vector(s): {', '.join(zeros[:3])} — text may be "
                             "whitespace-only or the model failed to load")

    if cfg.embedding.normalize:
        worst = float(np.max(np.abs(norms - 1.0)))
        if worst > NORM_TOLERANCE:
            raise EmbeddingError(
                f"vectors are not unit length (max |‖v‖-1| = {worst:.4f}); cosine "
                "distance would be meaningless")
    else:
        worst = float(np.max(np.abs(norms - 1.0)))

    return {
        "norms_min": float(norms.min()),
        "norms_max": float(norms.max()),
        "norm_max_deviation": worst,
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "dtype": str(vectors.dtype),
    }


def ids_digest(chunks: Sequence[Chunk]) -> str:
    """Order-sensitive digest of the id column: proves .npy rows map to chunks."""
    raw = "\n".join(c.chunk_id for c in chunks).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def probe(vectors: np.ndarray, chunks: Sequence[Chunk], qvec: np.ndarray) -> tuple[Chunk, float]:
    scores = vectors @ qvec.astype(np.float32)
    best = int(np.argmax(scores))
    return chunks[best], float(scores[best])


# ------------------------------------------------------------------ artifacts

def sample_markdown(chunks: Sequence[Chunk], vectors: np.ndarray,
                    probes: list[dict], stats: dict, cfg: AppConfig,
                    ids_sha: str) -> str:
    lines = [
        "# Embedding sample",
        "",
        f"- model: `{cfg.embedding.model_id}`",
        f"- shape: **{vectors.shape[0]} × {vectors.shape[1]}** `{vectors.dtype}`",
        f"- normalised: **{cfg.embedding.normalize}** "
        f"(‖v‖ range {stats['norms_min']:.6f}–{stats['norms_max']:.6f})",
        f"- fingerprint: `{cfg.fingerprint}` · ids_sha256 `{ids_sha[:16]}…`",
        f"- preprocessing: `{cfg.embedding.preprocessing}`",
        "",
        f"## First {SAMPLE_DIMS} dimensions of {min(SAMPLE_CHUNKS, len(chunks))} chunks",
        "",
        "The full matrix is `embeddings.npy`; this table exists so the numbers are",
        "visible in a diff without numpy.",
        "",
        "| chunk_id | fact_keys | ‖v‖ | dims 0–7 |",
        "|---|---|---|---|",
    ]
    # Prefer fact-bearing chunks across different documents: they are the ones
    # whose vectors will actually be cited.
    chosen: list[int] = []
    seen_docs: set[str] = set()
    for i, chunk in enumerate(chunks):
        if chunk.is_atomic and chunk.fact_keys and chunk.doc_id not in seen_docs:
            chosen.append(i)
            seen_docs.add(chunk.doc_id)
        if len(chosen) >= SAMPLE_CHUNKS:
            break
    for i in range(len(chunks)):
        if len(chosen) >= SAMPLE_CHUNKS:
            break
        if i not in chosen:
            chosen.append(i)

    for i in chosen:
        vector = vectors[i]
        cells = ", ".join(f"{value:+.4f}" for value in vector[:SAMPLE_DIMS])
        facts = ", ".join(chunks[i].fact_keys) or "—"
        lines.append(f"| `{chunks[i].chunk_id}` | {facts} | "
                     f"{float(np.linalg.norm(vector)):.4f} | {cells} |")

    lines += [
        "",
        "## Semantic sanity probes",
        "",
        "A randomly initialised or broken model still returns 384 numbers, so the",
        "stage checks that four obvious queries land on a chunk that actually holds",
        "the answer. This is a smoke test, not the retrieval evaluation (§6).",
        "",
        "| query | expected | top-1 chunk | score | ✓ |",
        "|---|---|---|---|---|",
    ]
    for hit in probes:
        lines.append(f"| {hit['query']} | {hit['expected']} | `{hit['chunk_id']}` "
                     f"| {hit['score']:.3f} | {'✓' if hit['ok'] else '✗'} |")

    misses = [p for p in probes if not p["ok"]]
    if misses:
        lines += ["", f"> {len(misses)} probe(s) missed. Vectors are well-formed but "
                  "the model may be the wrong one — check `embedding.model_id`."]
    return "\n".join(lines)


def run_probes(embedder: Embedder, chunks: Sequence[Chunk],
               vectors: np.ndarray) -> list[dict]:
    queries = [q for q, _, _ in PROBES if q]
    qvectors = embedder.encode(queries)
    out = []
    for (query, doc_id, fact_key), qvec in zip(PROBES, qvectors):
        best, score = probe(vectors, chunks, qvec)
        ok = best.doc_id == doc_id and fact_key in best.fact_keys
        out.append({"query": query, "expected": f"{doc_id} · {fact_key}",
                    "chunk_id": best.chunk_id, "score": round(score, 3), "ok": ok})
    return out


# ------------------------------------------------------------------ the stage

@dataclass
class EmbeddingRun:
    chunks: list[Chunk]
    vectors: np.ndarray
    manifest: dict
    stats: dict


def embed_all(cfg: AppConfig, rec: StageRecorder, chunks: list[Chunk] | None = None,
              embedder: Embedder | None = None) -> EmbeddingRun:
    chunks = list(chunks) if chunks is not None else load_chunks(cfg)
    if not chunks:
        raise EmbeddingError("no chunks to embed")

    embedder = embedder or Embedder(cfg)
    vectors = embedder.encode([c.embed_text for c in chunks])
    stats = validate_vectors(vectors, chunks, cfg)

    probes = run_probes(embedder, chunks, vectors)
    if not any(p["ok"] for p in probes):
        rec.warnings.append("all semantic probes missed — vectors are well-formed but "
                            "the model looks wrong; check embedding.model_id")
    misses = sum(1 for p in probes if not p["ok"])
    if misses and any(p["ok"] for p in probes):
        rec.warnings.append(f"{misses}/{len(probes)} semantic probes missed; "
                            "see 03_embeddings/sample_dims.md")

    ids_sha = ids_digest(chunks)
    chunk_run = _chunk_run_id(cfg)
    manifest = {
        "run_id": rec.run_id,
        "model_id": cfg.embedding.model_id,
        "dim": cfg.embedding.dim,
        "normalized": cfg.embedding.normalize,
        "max_seq_length": cfg.embedding.max_seq_length,
        "batch_size": cfg.embedding.batch_size,
        "device": cfg.embedding.device,
        "preprocessing_version": cfg.embedding.preprocessing,
        "fingerprint": cfg.fingerprint,
        "collection": cfg.collection_name,
        "count": len(chunks),
        "documents": len({c.doc_id for c in chunks}),
        "chunk_run_id": chunk_run,
        "max_chunk_tokens": max(c.token_count for c in chunks),
        "ids_sha256": ids_sha,
        "vector_dtype": stats["dtype"],
        "norm_max_deviation": round(stats["norm_max_deviation"], 8),
        "probes_passed": sum(1 for p in probes if p["ok"]),
        "probes_total": len(probes),
        "created_at": _now(),
    }

    rec.npy("03_embeddings", "embeddings.npy", vectors)
    rec.json("03_embeddings", "embedding_manifest.json", manifest)
    rec.markdown("03_embeddings", "sample_dims.md",
                 sample_markdown(chunks, vectors, probes, stats, cfg, ids_sha))
    rec.manifest("03_embeddings", {
        "embeddings": len(chunks),
        "dim": cfg.embedding.dim,
        "model_id": cfg.embedding.model_id,
        "fingerprint": cfg.fingerprint,
        "ids_sha256": ids_sha,
        "chunk_run_id": chunk_run,
        "max_chunk_tokens": manifest["max_chunk_tokens"],
        "probes_passed": manifest["probes_passed"],
        "probes_total": manifest["probes_total"],
    })
    return EmbeddingRun(chunks=list(chunks), vectors=vectors, manifest=manifest, stats=stats)


def _chunk_run_id(cfg: AppConfig) -> str:
    path = cfg.stage_dir("02_chunks") / "stage_manifest.json"
    try:
        return json.loads(path.read_text("utf-8")).get("run_id", "")
    except (OSError, ValueError):
        return ""


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def embedder_check(cfg: AppConfig) -> dict[str, object]:
    """Cheap `--check-embeddings`-style summary used by the pipeline/UI later."""
    path = cfg.stage_dir("03_embeddings") / "embedding_manifest.json"
    if not path.exists():
        raise PipelineError(f"{path} is missing — run `--stage embed`")
    manifest = json.loads(path.read_text("utf-8"))
    if manifest.get("fingerprint") != cfg.fingerprint:
        raise EmbeddingError(
            f"embeddings were built with fingerprint {manifest.get('fingerprint')} but "
            f"config gives {cfg.fingerprint} — re-run `--stage embed` (and `--stage store`)")
    vectors = np.load(cfg.stage_dir("03_embeddings") / "embeddings.npy",
                      allow_pickle=False)
    if vectors.shape != (manifest["count"], manifest["dim"]):
        raise EmbeddingError(f"embeddings.npy shape {vectors.shape} does not match "
                             f"manifest {manifest['count']}×{manifest['dim']}")
    if float(np.linalg.norm(vectors[0])) < 1e-8:
        raise EmbeddingError("first vector is zero")
    return {"manifest": manifest, "shape": tuple(vectors.shape)}

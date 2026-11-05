"""Vector DB stage: Chroma collection bound to the embedding fingerprint.

`implementation.md` §5. The store deliberately owns *no* vectors of its own: it
reads `03_embeddings/` (matrix + manifest) and `02_chunks/` (display text and
metadata) and refuses to combine them if they disagree.

The fingerprint is the coupling point between phases 4 and 5:

    fingerprint = sha256(model_id | dim | normalize | preprocessing)[:12]
    collection  = f"mf_faq_{model_slug}_{fingerprint}"

A mismatched collection raises `EmbeddingMismatchError` instead of silently
searching a different vector space — the classic bug this design exists to kill.

Outputs: `04_vectordb/collection_report.md`, `collection_manifest.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from config.config import AppConfig, ConfigError
from src.artifacts import StageRecorder
from src.embedder import chunks_path, embedder_check, ids_digest, load_chunks
from src.loader import PipelineError
from src.models import Chunk

BATCH_SIZE = 256
COSINE_TOLERANCE = 1e-4


class StoreError(RuntimeError):
    """The collection and the config disagree, or the write did not land."""


class EmbeddingMismatchError(StoreError):
    """A collection was built with a different embedding model/preprocessing."""


# ------------------------------------------------------------------ metadata

def chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    """Scalar-only metadata (Chroma rejects lists/None). Keep `chunks.jsonl` rich."""
    return {
        "doc_id": chunk.doc_id,
        "source_url": chunk.meta.source_url,
        "page_title": chunk.meta.page_title,
        "page_type": chunk.meta.page_type,
        "scheme": chunk.meta.scheme or "",
        "plan": chunk.meta.plan,
        "amc": chunk.meta.amc,
        "section": " > ".join(chunk.section_path),
        "kind": chunk.kind,
        "fact_keys": ",".join(chunk.fact_keys),
        "is_quarantined": bool(chunk.is_quarantined),
        "fetched_at": chunk.meta.fetched_at.isoformat(),
        "chunk_strategy": chunk.chunk_strategy,
        "content_sha256": chunk.content_sha256,
        "token_count": int(chunk.token_count),
    }


# ------------------------------------------------------------------ the store

@dataclass
class StoreStatus:
    name: str
    fingerprint: str
    dim: int
    distance: str
    count: int
    ids_sha256: str
    per_doc: dict[str, int]
    stale_deleted: int
    rebuilt: bool
    min_cosine: float
    nearest_is_self: bool
    samples: list[dict[str, Any]]


class Store:
    """Minimal Chroma wrapper; no embedding is done here, ever."""

    def __init__(self, cfg: AppConfig, client=None) -> None:
        self.cfg = cfg
        self.path = cfg.resolve(cfg.vectorstore.persist_dir)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as exc:                   # noqa: BLE001
                raise ConfigError(
                    "chromadb is not installed — run `pip install -r requirements.txt`"
                ) from exc
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
        return self._client

    @property
    def chroma_version(self) -> str:
        try:
            import chromadb
            return chromadb.__version__
        except Exception:                            # noqa: BLE001
            return "unknown"

    def names(self) -> list[str]:
        collections = self.client.list_collections()
        # 0.5.x returns Collection objects, older/newer versions return names.
        return [c if isinstance(c, str) else c.name for c in collections]

    def ensure(self, name: str | None = None, fingerprint: str | None = None,
               dim: int | None = None, rebuild: bool = False):
        """Create or open the collection, failing loudly on fingerprint drift."""
        name = name or self.cfg.collection_name
        fingerprint = fingerprint or self.cfg.fingerprint
        dim = dim or self.cfg.embedding.dim
        exists = name in self.names()

        if exists and rebuild:
            self.client.delete_collection(name)
            exists = False

        if exists:
            collection = self.client.get_collection(name)
            metadata = collection.metadata or {}
            stored = metadata.get("fingerprint")
            if stored != fingerprint:
                raise EmbeddingMismatchError(
                    f"collection '{name}' was built with fingerprint {stored} but the "
                    f"config gives {fingerprint} ({self.cfg.embedding.model_id}, "
                    f"{dim}-d, {self.cfg.embedding.preprocessing}). Re-run:\n"
                    f"  python -m app.pipeline --stage store --rebuild")
            if metadata.get("dim") not in (None, dim):
                raise EmbeddingMismatchError(
                    f"collection '{name}' stores {metadata.get('dim')}-d vectors but "
                    f"config.embedding.dim={dim} — run `--stage store --rebuild`")
            return collection

        return self.client.get_or_create_collection(
            name=name,
            metadata={
                "hnsw:space": self.cfg.vectorstore.distance,
                "fingerprint": fingerprint,
                "dim": int(dim),
                "model_id": self.cfg.embedding.model_id,
                "preprocessing": self.cfg.embedding.preprocessing,
            },
        )

    # ---- writes ---------------------------------------------------------

    def sync(self, collection, chunks: Sequence[Chunk],
             vectors: np.ndarray) -> int:
        """Upsert all chunks, deleting ids that no longer exist. Returns stale count."""
        expected = {c.chunk_id for c in chunks}
        existing = set(collection.get(include=["metadatas"])["ids"])
        stale = sorted(existing - expected)
        if stale:
            for batch in _batches(stale, BATCH_SIZE):
                collection.delete(ids=batch)

        for batch in _batches(list(range(len(chunks))), BATCH_SIZE):
            collection.upsert(
                ids=[chunks[i].chunk_id for i in batch],
                embeddings=[vectors[i].tolist() for i in batch],
                documents=[chunks[i].text for i in batch],       # display text only
                metadatas=[chunk_metadata(chunks[i]) for i in batch],
            )
        return len(stale)

    # ---- reads ----------------------------------------------------------

    def fetch(self, collection, ids: Sequence[str] | None = None) -> dict:
        if ids is None:
            return collection.get(include=["documents", "metadatas", "embeddings"])
        out: dict[str, list] = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        for batch in _batches(list(ids), BATCH_SIZE):
            part = collection.get(ids=batch,
                                  include=["documents", "metadatas", "embeddings"])
            for key in out:
                out[key].extend(part[key])
        return out

    def query(self, collection, vector: np.ndarray, k: int = 3,
              where: dict | None = None) -> dict:
        return collection.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
            n_results=k, where=where or None,
            include=["documents", "metadatas", "distances"],
        )


def _batches(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


# ------------------------------------------------------------------ checks

def _load_vectors(cfg: AppConfig, chunks: Sequence[Chunk]) -> tuple[np.ndarray, dict]:
    """Read `embeddings.npy` + manifest and prove they still match the chunks."""
    checked = embedder_check(cfg)                    # fingerprint + shape
    manifest = checked["manifest"]
    vectors = np.load(cfg.stage_dir("03_embeddings") / "embeddings.npy",
                      allow_pickle=False)
    if len(chunks) != manifest["count"]:
        raise StoreError(
            f"chunks.jsonl has {len(chunks)} chunks but embeddings.npy has "
            f"{manifest['count']} rows — re-run `--stage chunk` then `--stage embed`")
    digest = ids_digest(chunks)
    if digest != manifest["ids_sha256"]:
        raise StoreError(
            "chunk ids no longer match the embedding manifest (ids_sha256 differs) — "
            "re-run `--stage chunk` then `--stage embed`")
    if vectors.dtype != np.float32 or vectors.shape != (len(chunks), cfg.embedding.dim):
        raise StoreError(f"embeddings.npy is {vectors.dtype} {vectors.shape}, expected "
                         f"float32 {(len(chunks), cfg.embedding.dim)}")
    return vectors, manifest


def verify(collection, chunks: Sequence[Chunk], vectors: np.ndarray,
           cfg: AppConfig) -> dict[str, Any]:
    """Read the collection back and compare it with what we meant to write."""
    stored = collection.count()
    if stored != len(chunks):
        raise StoreError(f"collection holds {stored} records, expected {len(chunks)}")

    fetched = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = fetched["ids"]
    if len(ids) != len(set(ids)):
        raise StoreError("collection contains duplicate ids")
    if set(ids) != {c.chunk_id for c in chunks}:
        missing = {c.chunk_id for c in chunks} - set(ids)
        raise StoreError(f"collection is missing {len(missing)} chunk(s), "
                         f"e.g. {sorted(missing)[:3]}")

    by_id = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    index_of = {chunk_id: by_id[chunk_id] for chunk_id in ids}
    stored_vectors = np.asarray(fetched["embeddings"], dtype=np.float32)
    expected_vectors = np.vstack([vectors[index_of[chunk_id]] for chunk_id in ids])

    cosines = np.sum(stored_vectors * expected_vectors, axis=1)
    worst = float(np.min(cosines))
    if worst < 1 - COSINE_TOLERANCE:
        bad = ids[int(np.argmin(cosines))]
        raise StoreError(f"stored vector for {bad} does not match embeddings.npy "
                         f"(cosine {worst:.6f}) — the matrix rows and ids are misaligned")

    for chunk_id, document, metadata in zip(ids, fetched["documents"],
                                            fetched["metadatas"]):
        chunk = chunks[by_id[chunk_id]]
        if document != chunk.text:
            raise StoreError(f"{chunk_id}: stored document differs from chunk.text")
        if metadata.get("fact_keys") != ",".join(chunk.fact_keys):
            raise StoreError(f"{chunk_id}: metadata fact_keys drifted")
        if metadata.get("is_quarantined") != bool(chunk.is_quarantined):
            raise StoreError(f"{chunk_id}: metadata is_quarantined drifted")
        if not isinstance(metadata.get("token_count"), int):
            raise StoreError(f"{chunk_id}: metadata must be scalar (token_count is "
                             f"{type(metadata.get('token_count')).__name__})")

    # Distance sanity: a chunk's own vector must be its own nearest neighbour.
    probe_index = next((i for i, c in enumerate(chunks) if c.fact_keys), 0)
    probe = chunks[probe_index]
    result = collection.query(query_embeddings=[vectors[probe_index].tolist()],
                              n_results=1, include=["distances"])
    nearest = result["ids"][0][0]
    score = 1 - float(result["distances"][0][0]) if cfg.vectorstore.distance == "cosine" \
        else float(result["distances"][0][0])
    if nearest != probe.chunk_id:
        raise StoreError(f"nearest neighbour of {probe.chunk_id} is {nearest}; the index "
                         "or distance space is wrong")

    return {
        "count": int(stored),
        "min_cosine": worst,
        "nearest_is_self": True,
        "probe": {"chunk_id": probe.chunk_id, "score": round(score, 6)},
        "ids_sha256": ids_digest(chunks),
    }


# ------------------------------------------------------------------ artifacts

def _counts_per_doc(chunks: Sequence[Chunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
    return dict(sorted(counts.items()))


def _samples(collection, chunks: Sequence[Chunk],
             count: int = 3) -> list[dict[str, Any]]:
    picks = [c for c in chunks if c.is_atomic and c.fact_keys][:count]
    if len(picks) < count:
        picks += [c for c in chunks if c not in picks][:count - len(picks)]
    fetched = collection.get(ids=[c.chunk_id for c in picks],
                             include=["documents", "metadatas"])
    by_id = {chunk_id: (document, metadata) for chunk_id, document, metadata
             in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])}
    out = []
    for chunk in picks:
        document, metadata = by_id.get(chunk.chunk_id, (chunk.text, chunk_metadata(chunk)))
        out.append({
            "chunk_id": chunk.chunk_id,
            "section": metadata.get("section"),
            "scheme": metadata.get("scheme") or "—",
            "fact_keys": metadata.get("fact_keys") or "—",
            "token_count": metadata.get("token_count"),
            "text": document,
        })
    return out


def report_markdown(cfg: AppConfig, status: StoreStatus, manifest: dict,
                    chunks: Sequence[Chunk]) -> str:
    lines = [
        "# Collection report",
        "",
        f"- collection: `{status.name}`",
        f"- distance: **{status.distance}** · vectors: **{status.dim}-d** "
        f"(`{manifest['model_id']}`)",
        f"- fingerprint: `{status.fingerprint}`",
        f"- records: **{status.count}** (chunks.jsonl: {len(chunks)})",
        f"- ids_sha256: `{status.ids_sha256[:16]}…`",
        f"- chromadb: `{manifest['chromadb_version']}` · persist: `{cfg.vectorstore.persist_dir}`",
        "",
        "## Checks",
        "",
        f"- [{'x' if status.count == len(chunks) else ' '}] "
        f"chroma count equals chunk count ({status.count} = {len(chunks)})",
        f"- [{'x' if status.min_cosine >= 1 - COSINE_TOLERANCE else ' '}] "
        f"stored vectors match embeddings.npy (min cosine {status.min_cosine:.8f})",
        f"- [{'x' if status.nearest_is_self else ' '}] "
        f"a chunk is its own nearest neighbour "
        f"(`{manifest['nearest_probe']['chunk_id']}`, score "
        f"{manifest['nearest_probe']['score']:.4f})",
        f"- [x] metadata is scalar; documents are `chunk.text` (not `embed_text`)",
        f"- [x] fingerprint in collection metadata matches config (`{status.fingerprint}`)",
        "",
        "## Records per document",
        "",
        "| doc_id | chunks |",
        "|---|---|",
    ]
    lines += [f"| {doc_id} | {count} |" for doc_id, count in status.per_doc.items()]
    lines += [
        f"| **TOTAL** | **{sum(status.per_doc.values())}** |",
        "",
        f"Stale ids removed this run: **{status.stale_deleted}**"
        f"{' · collection rebuilt' if status.rebuilt else ''}",
        "",
        "## Sample records",
        "",
    ]
    for sample in status.samples:
        lines += [
            f"### `{sample['chunk_id']}` · {sample['scheme']} · {sample['fact_keys']}",
            "",
            f"_{sample['section']}_ · {sample['token_count']} tokens",
            "",
            "```text",
            sample["text"],
            "```",
            "",
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------ the stage

@dataclass
class StoreRun:
    status: StoreStatus
    collection: Any


def store_all(cfg: AppConfig, rec: StageRecorder, *, rebuild: bool = False,
              chunks: list[Chunk] | None = None, vectors: np.ndarray | None = None,
              store: Store | None = None) -> StoreRun:
    chunks = list(chunks) if chunks is not None else load_chunks(cfg)
    if not chunks:
        raise StoreError("nothing to store")
    if vectors is None:
        vectors, _manifest = _load_vectors(cfg, chunks)
    else:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), cfg.embedding.dim):
            raise StoreError(f"vectors are {vectors.shape}, expected "
                             f"{(len(chunks), cfg.embedding.dim)}")

    store = store or Store(cfg)
    collection = store.ensure(rebuild=rebuild)
    stale = store.sync(collection, chunks, vectors)
    verified = verify(collection, chunks, vectors, cfg)

    status = StoreStatus(
        name=cfg.collection_name,
        fingerprint=cfg.fingerprint,
        dim=cfg.embedding.dim,
        distance=cfg.vectorstore.distance,
        count=verified["count"],
        ids_sha256=verified["ids_sha256"],
        per_doc=_counts_per_doc(chunks),
        stale_deleted=stale,
        rebuilt=rebuild,
        min_cosine=verified["min_cosine"],
        nearest_is_self=verified["nearest_is_self"],
        samples=_samples(collection, chunks),
    )

    manifest = {
        "run_id": rec.run_id,
        "collection": status.name,
        "provider": cfg.vectorstore.provider,
        "persist_dir": str(cfg.resolve(cfg.vectorstore.persist_dir)),
        "distance": status.distance,
        "dim": status.dim,
        "model_id": cfg.embedding.model_id,
        "fingerprint": status.fingerprint,
        "count": status.count,
        "chunk_count": len(chunks),
        "per_doc": status.per_doc,
        "ids_sha256": status.ids_sha256,
        "min_cosine": round(status.min_cosine, 8),
        "nearest_probe": verified["probe"],
        "stale_deleted": status.stale_deleted,
        "rebuilt": status.rebuilt,
        "chromadb_version": store.chroma_version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    rec.json("04_vectordb", "collection_manifest.json", manifest)
    rec.markdown("04_vectordb", "collection_report.md",
                 report_markdown(cfg, status, manifest, chunks))
    rec.manifest("04_vectordb", {
        "collection": status.name,
        "records": status.count,
        "chunk_count": len(chunks),
        "dim": status.dim,
        "fingerprint": status.fingerprint,
        "ids_sha256": status.ids_sha256,
        "stale_deleted": status.stale_deleted,
        "rebuilt": status.rebuilt,
        "min_cosine": round(status.min_cosine, 8),
        "chromadb_version": store.chroma_version,
    })
    return StoreRun(status=status, collection=collection)


def open_collection(cfg: AppConfig, store: Store | None = None):
    """§6 entry point: open the index, or explain exactly why you can't."""
    store = store or Store(cfg)
    name = cfg.collection_name
    if name not in store.names():
        raise StoreError(
            f"collection '{name}' does not exist — run `python -m app.pipeline "
            f"--stage store` first")
    collection = store.client.get_collection(name)
    metadata = collection.metadata or {}
    if metadata.get("fingerprint") != cfg.fingerprint:
        raise EmbeddingMismatchError(
            f"collection '{name}' was built with fingerprint "
            f"{metadata.get('fingerprint')} but config gives {cfg.fingerprint} — "
            f"run `--stage store --rebuild`")
    if collection.count() == 0:
        raise StoreError(f"collection '{name}' is empty — re-run `--stage store`")
    return collection


def collection_exists(cfg: AppConfig) -> bool:
    try:
        return cfg.collection_name in Store(cfg).names()
    except ConfigError:
        return False

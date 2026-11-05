"""Structure-aware chunking with atomic facts (implementation.md §3).

A fact never leaves its label. Fixed-window splitting is wrong for this corpus:
it separates "Exit load" from "Nil", which is the one failure this product
cannot afford. So blocks are classified first — atomic units pass through alone,
prose is merged and split on sentence boundaries under a hard token cap.

Every chunk is written to `02_chunks/` so the step is inspectable:
`chunks.jsonl`, `chunks_preview.md`, `chunk_stats.csv`.
"""

from __future__ import annotations

import math
import re
from hashlib import sha256
from statistics import mean
from typing import Iterable, Iterator

from config.config import AppConfig
from src.artifacts import StageRecorder
from src.lexicon import Lexicon, load_lexicon
from src.loader import PipelineError, load_documents
from src.models import Block, Chunk, Document
from src.tokenization import TokenCounter

CHUNK_STRATEGY = "structure_aware_v1"
ATOMIC_SPLIT_STRATEGY = "atomic_split_v1"

#: Kinds that are facts by construction.
ATOMIC_KINDS = {"table_row", "list_item", "definition"}

#: And blocks that are facts by content, whatever their kind.
ATOMIC_TERMS = re.compile(r"\b(lock[- ]?in|riskometer)\b", re.I)

#: "Minimum SIP: ₹500" — a labelled value that must not be split from its label.
LABELED = re.compile(r"^[A-Z][A-Za-z0-9 /()%.,&+\-–]{2,60}\s*[:\-–]\s*\S")

SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
WHITESPACE = re.compile(r"[ \t]+")


class ChunkError(RuntimeError):
    """A chunk violated a hard invariant (cap or id)."""


# ------------------------------------------------------------------ classify

def is_atomic(block: Block) -> bool:
    if block.kind in ATOMIC_KINDS:
        return True
    if ATOMIC_TERMS.search(block.text):
        return True
    return bool(LABELED.match(block.text.strip()))


def render_table_row(block: Block) -> str:
    """Keep a table row readable standalone by prefixing its real header row."""
    if block.rows:
        header, cells = block.rows[0], block.rows[1:] or [[]]
        first_cell = cells[0][:1] if cells else []
        # Headerless label/value tables are already self-contained ("AUM: ₹…").
        if len(header) >= 2 and header != first_cell:
            return f"Columns: {' | '.join(header)}\n{block.text}"
    return block.text


# ------------------------------------------------------------------ tokenize

def _sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_END.split(text) if part.strip()]


def _fit(sentences: Iterable[str], limit: int, counter: TokenCounter) -> list[str]:
    """Split any single sentence longer than `limit` on word boundaries."""
    fitted: list[str] = []
    for sentence in sentences:
        if counter.count(sentence) <= limit:
            fitted.append(sentence)
            continue
        words, current, total = sentence.split(), [], 0
        for word in words:
            cost = counter.count(word) + 1
            if total + cost > limit and current:
                fitted.append(" ".join(current))
                current, total = [word], counter.count(word)
            else:
                current.append(word)
                total += cost
        if current:
            fitted.append(" ".join(current))
    return fitted


def _windows(sentences: list[str], limit: int, overlap_tokens: int,
             counter: TokenCounter) -> Iterator[list[str]]:
    """Sliding sentence windows of <= `limit` tokens, with prose-only overlap."""
    i, n = 0, len(sentences)
    while i < n:
        j, total = i, 0
        while j < n:
            cost = counter.count(sentences[j]) + (1 if j > i else 0)
            if total + cost > limit and j > i:
                break
            total += cost
            j += 1
        yield sentences[i:j]
        if j >= n:
            break
        k, overlap = j, 0
        while k > i + 1 and overlap + counter.count(sentences[k - 1]) <= overlap_tokens:
            k -= 1
            overlap += counter.count(sentences[k])
        i = k if k > i else j                              # always make progress


# ------------------------------------------------------------------ assembly

def _prefix(doc: Document, section: list[str]) -> str:
    return f"{doc.meta.page_title} | {' > '.join(section)}"


def _embed_text(doc: Document, section: list[str], text: str,
                counter: TokenCounter, max_tokens: int) -> tuple[str, bool]:
    full = f"{_prefix(doc, section)} | {text}"
    if counter.count(full) <= max_tokens:
        return full, True
    # Never let a long page title eat the fact: drop the title, then the section.
    short = f"{' > '.join(section)} | {text}"
    if counter.count(short) <= max_tokens:
        return short, False
    return text, False


def _quarantine(text: str, patterns: list[re.Pattern]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _chunk(doc: Document, text: str, section: list[str], kind: str, is_atomic: bool,
           counter: TokenCounter, lex: Lexicon, quarantine: list[re.Pattern],
           cfg: AppConfig, run_id: str) -> tuple[Chunk, bool]:
    clean = WHITESPACE.sub(" ", text).strip()
    embed, full_prefix = _embed_text(doc, section, clean, counter, cfg.chunking.max_tokens)
    return Chunk(
        chunk_id="",
        doc_id=doc.meta.doc_id,
        text=clean,
        embed_text=embed,
        section_path=list(section or ["Page"]),
        kind=kind,
        fact_keys=lex.fact_keys(clean),
        is_atomic=is_atomic,
        is_quarantined=_quarantine(clean, quarantine),
        token_count=counter.count(clean),
        meta=doc.meta,
        chunk_strategy=CHUNK_STRATEGY if full_prefix else f"{CHUNK_STRATEGY}+prefix_trimmed",
        content_sha256=doc.content_sha256,
        run_id=run_id,
    ), full_prefix


def chunk_document(doc: Document, cfg: AppConfig, counter: TokenCounter,
                   lex: Lexicon, run_id: str = "") -> tuple[list[Chunk], list[str]]:
    """Chunk one document. Returns (chunks, warnings) — warnings are never hidden."""
    max_tokens = cfg.chunking.max_tokens
    overlap = int(max_tokens * cfg.chunking.overlap_ratio)
    quarantine = [re.compile(p, re.I) for p in cfg.chunking.quarantine_patterns]
    warnings: list[str] = []
    out: list[Chunk] = []

    def emit(text: str, section: list[str], kind: str, atomic: bool) -> None:
        chunk, full_prefix = _chunk(doc, text, section, kind, atomic,
                                    counter, lex, quarantine, cfg, run_id)
        if not full_prefix:
            warnings.append(f"{doc.meta.doc_id}: embed prefix trimmed for a chunk in "
                            f"{' > '.join(section) or 'Page'}")
        out.append(chunk)

    def prose_budget(section: list[str]) -> int:
        prefix_tokens = counter.count(_prefix(doc, section) + " | ")
        return max(64, max_tokens - prefix_tokens)

    buffer: list[Block] = []
    buffer_section: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        section = buffer_section or ["Page"]
        text = "\n".join(render_table_row(b) if b.kind == "table_row" else b.text
                         for b in buffer)
        limit = prose_budget(section)
        sentences = _fit(_sentences(text), limit, counter)
        for window in _windows(sentences, limit, overlap, counter):
            emit(" ".join(window), section, buffer[0].kind, False)
        buffer.clear()

    for block in doc.blocks:
        if not block.text.strip():
            continue
        if block.kind == "heading":
            flush()                              # headings set section paths, never facts
            continue
        if is_atomic(block):
            flush()
            text = render_table_row(block) if block.kind == "table_row" else block.text
            section = block.section_path or ["Page"]
            if counter.count(text) <= max_tokens:
                emit(text, section, block.kind, True)
            else:
                warnings.append(f"{doc.meta.doc_id}: atomic block over {max_tokens} tokens "
                                f"({counter.count(text)}) split on sentence boundaries")
                for window in _windows(_fit(_sentences(text), max_tokens, counter),
                                       max_tokens, 0, counter):
                    emit(" ".join(window), section, block.kind, False)
            continue

        section = block.section_path or ["Page"]
        if buffer and section != buffer_section:
            flush()
        candidate = "\n".join([*(b.text for b in buffer), block.text])
        if buffer and counter.count(candidate) > cfg.chunking.merge_target_tokens[1]:
            flush()
        if not buffer:
            buffer_section = section
        buffer.append(block)
    flush()

    # Dedupe identical text (repeated nav blocks, repeated facts across sections).
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for chunk in out:
        key = sha256(re.sub(r"[^a-z0-9]+", "", chunk.text.lower()).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)

    for ordinal, chunk in enumerate(deduped):
        chunk.chunk_id = f"{doc.meta.doc_id}#{ordinal:02d}"
        if chunk.token_count > max_tokens:
            raise ChunkError(f"{chunk.chunk_id} has {chunk.token_count} tokens "
                             f"(limit {max_tokens}) — refusing to write a silently truncatable chunk")
    return deduped, warnings


# ------------------------------------------------------------------ the stage

def chunk_all(cfg: AppConfig, rec: StageRecorder, only: str | None = None) -> list[Chunk]:
    if only:
        # A partial chunk run would rewrite the one combined chunks.jsonl and
        # silently drop every other document from the next stage's input.
        raise PipelineError(
            "--only is not supported for --stage chunk (it rewrites chunks.jsonl for "
            "the whole corpus); restrict --stage load instead")
    docs = load_documents(cfg)

    counter = TokenCounter(cfg)
    rec.warnings.extend(counter.notes)
    lex = load_lexicon(cfg.resolve(cfg.corpus.lexicon))

    all_chunks: list[Chunk] = []
    per_doc: list[dict] = []
    for doc in docs:
        chunks, warnings = chunk_document(doc, cfg, counter, lex, run_id=rec.run_id)
        rec.warnings.extend(warnings)
        all_chunks.extend(chunks)
        tokens = [c.token_count for c in chunks]
        atomic = sum(1 for c in chunks if c.is_atomic)
        per_doc.append({
            "doc_id": doc.meta.doc_id,
            "chunks": len(chunks),
            "tokens_mean": round(mean(tokens), 1) if tokens else 0,
            "tokens_p95": _p95(tokens),
            "tokens_max": max(tokens, default=0),
            "atomic": atomic,
            "atomic_pct": round(100 * atomic / len(chunks), 1) if chunks else 0,
            "quarantined": sum(1 for c in chunks if c.is_quarantined),
        })
        print(f"  {doc.meta.doc_id:22} {len(chunks):3} chunks  "
              f"{atomic:3} atomic  max {max(tokens, default=0):3} tok")

    over = [c for c in all_chunks if c.token_count > cfg.chunking.max_tokens]
    if over:  # defence in depth; chunk_document already refuses these
        raise ChunkError(f"{len(over)} chunks exceed {cfg.chunking.max_tokens} tokens")

    stats = per_doc + [_total_row(all_chunks)]
    rec.jsonl("02_chunks", "chunks.jsonl", [c.to_dict() for c in all_chunks])
    rec.csv("02_chunks", "chunk_stats.csv", stats,
            columns=["doc_id", "chunks", "tokens_mean", "tokens_p95", "tokens_max",
                     "atomic", "atomic_pct", "quarantined"])
    rec.markdown("02_chunks", "chunks_preview.md",
                 preview(all_chunks, stats, cfg, counter))
    rec.manifest("02_chunks", {
        "documents": len(docs),
        "chunks": len(all_chunks),
        "atomic": sum(1 for c in all_chunks if c.is_atomic),
        "quarantined": sum(1 for c in all_chunks if c.is_quarantined),
        "tokens_total": sum(c.token_count for c in all_chunks),
        "tokens_max": max((c.token_count for c in all_chunks), default=0),
        "token_limit": cfg.chunking.max_tokens,
        "over_limit": len(over),
        "tokenizer": counter.describe(),
    })
    return all_chunks


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _total_row(chunks: list[Chunk]) -> dict:
    tokens = [c.token_count for c in chunks]
    total_chunks = len(chunks)
    total_atomic = sum(1 for c in chunks if c.is_atomic)
    return {
        "doc_id": "TOTAL",
        "chunks": total_chunks,
        "tokens_mean": round(mean(tokens), 1) if tokens else 0,
        "tokens_p95": _p95(tokens),
        "tokens_max": max(tokens, default=0),
        "atomic": total_atomic,
        "atomic_pct": round(100 * total_atomic / total_chunks, 1) if total_chunks else 0,
        "quarantined": sum(1 for c in chunks if c.is_quarantined),
    }


def lockin_chunk(chunks: list[Chunk]) -> Chunk | None:
    """The §3 done-check: ELSS lock-in and its duration in one chunk."""
    return next((c for c in chunks
                 if "lock-in" in c.text.lower() and "3 years" in c.text.lower()), None)


def preview(chunks: list[Chunk], stats: list[dict], cfg: AppConfig,
            counter: TokenCounter) -> str:
    lockin = lockin_chunk(chunks)
    lines = [
        "# Chunk preview",
        "",
        f"- documents: **{len({c.doc_id for c in chunks})}**",
        f"- chunks: **{len(chunks)}**",
        f"- token limit: **{cfg.chunking.max_tokens}** "
        f"(tokenizer: `{counter.mode}`)",
        f"- max chunk: **{max((c.token_count for c in chunks), default=0)} tokens**",
        f"- atomic: **{sum(1 for c in chunks if c.is_atomic)}**",
        f"- quarantined: **{sum(1 for c in chunks if c.is_quarantined)}**",
        "",
        "## Checks",
        "",
        f"- [{'x' if not any(c.token_count > cfg.chunking.max_tokens for c in chunks) else ' '}] "
        f"no chunk exceeds {cfg.chunking.max_tokens} tokens",
        f"- [{'x' if lockin else ' '}] ELSS lock-in and duration in one chunk"
        + (f" (`{lockin.chunk_id}`: “{lockin.text}”)" if lockin else ""),
        "",
        "## Stats",
        "",
        "| doc_id | chunks | mean | p95 | max | atomic % | quarantined |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {row['doc_id']} | {row['chunks']} | {row['tokens_mean']} | "
              f"{row['tokens_p95']} | {row['tokens_max']} | {row['atomic_pct']} | "
              f"{row['quarantined']} |" for row in stats]
    lines.append("")

    for chunk in chunks:
        facts = ", ".join(chunk.fact_keys) or "—"
        flags = " · atomic" if chunk.is_atomic else ""
        flags += " · quarantined" if chunk.is_quarantined else ""
        lines += [
            f"### `{chunk.chunk_id}` · {chunk.token_count} tok · {facts}{flags}",
            "",
            f"_{chunk.meta.page_title} — {' > '.join(chunk.section_path)}_",
            "",
            "```text",
            chunk.text,
            "```",
            "",
        ]
    return "\n".join(lines)

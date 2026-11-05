"""Chunker invariants (implementation.md §3.2). No network required."""

from __future__ import annotations

from datetime import date

import pytest

from config.config import PROJECT_ROOT, load_config
from src.chunker import chunk_document, lockin_chunk
from src.lexicon import load_lexicon
from src.models import Block, DocMeta, Document
from src.tokenization import TokenCounter


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def lex(cfg):
    return load_lexicon(cfg.resolve(cfg.corpus.lexicon))


@pytest.fixture(scope="module")
def counter(cfg):
    # allow_download=False keeps tests offline; exact if already cached.
    return TokenCounter(cfg, allow_download=False)


def make_doc(blocks: list[Block], doc_id: str = "test_doc", title: str = "HDFC Test Fund") -> Document:
    return Document(
        meta=DocMeta(doc_id=doc_id, source_url="https://groww.in/test", page_title=title,
                     page_type="groww_scheme", scheme="hdfc_elss", amc="HDFC Mutual Fund",
                     plan="direct_growth", fetched_at=date(2026, 11, 1)),
        blocks=blocks,
        content_sha256="abc123",
    )


def fact(label: str, value: str) -> Block:
    return Block(["Fund facts"], "definition", f"{label}: {value}")


def run(doc: Document, cfg, counter, lex):
    return chunk_document(doc, cfg, counter, lex, run_id="test")


def test_token_cap_including_embed_text(cfg, counter, lex):
    long_paragraph = " ".join(
        f"This is sentence {i} about the scheme and its expenses." for i in range(120))
    chunks, _ = run(make_doc([Block(["About"], "paragraph", long_paragraph)]),
                    cfg, counter, lex)

    assert len(chunks) > 1
    assert max(c.token_count for c in chunks) <= cfg.chunking.max_tokens
    assert max(counter.count(c.embed_text) for c in chunks) <= cfg.chunking.max_tokens


def test_prose_windows_overlap_on_a_sentence(cfg, counter, lex):
    paragraph = " ".join(
        f"Sentence {i} describes the scheme expense structure clearly." for i in range(40))
    chunks, _ = run(make_doc([Block(["About"], "paragraph", paragraph)]), cfg, counter, lex)

    assert len(chunks) > 1

    def sentences(text: str) -> set[str]:
        return {part.strip() for part in text.split(".") if part.strip()}

    # Overlap is what keeps a fact from falling into the seam between two windows.
    for previous, nxt in zip(chunks, chunks[1:]):
        assert sentences(previous.text) & sentences(nxt.text)


def test_atomic_facts_stay_whole_and_single_key(cfg, counter, lex):
    blocks = [
        fact("Expense ratio", "1.2% per annum"),
        fact("Exit load", "Nil"),
        fact("Lock-in", "3 years from the date of investment"),
        fact("Riskometer", "Very High Risk"),
        fact("Benchmark", "NIFTY 500 Total Return Index (NIFTY 500 TRI)"),
    ]
    chunks, _ = run(make_doc(blocks), cfg, counter, lex)

    assert len(chunks) == len(blocks)
    assert all(c.is_atomic for c in chunks)
    assert all(len(c.fact_keys) <= 1 for c in chunks if c.is_atomic)
    assert any(c.fact_keys == ["lock_in"] for c in chunks)
    assert lockin_chunk(chunks) is not None


def test_table_row_keeps_its_header(cfg, counter, lex):
    block = Block(["Table"], "table_row",
                  "Fund Name: HDFC Mid Cap Fund | Expense Ratio: 0.74",
                  rows=[["Fund Name", "Expense Ratio"], ["HDFC Mid Cap Fund", "0.74"]])
    chunks, _ = run(make_doc([block]), cfg, counter, lex)

    assert len(chunks) == 1
    assert "Columns: Fund Name | Expense Ratio" in chunks[0].text
    assert "Fund Name: HDFC Mid Cap Fund" in chunks[0].text


def test_disclaimer_is_quarantined_not_dropped(cfg, counter, lex):
    block = Block(["Disclaimer"], "paragraph",
                  "Mutual fund investments are subject to market risks. "
                  "Read all scheme related documents carefully before investing.")
    chunks, _ = run(make_doc([block]), cfg, counter, lex)

    assert chunks and all(c.is_quarantined for c in chunks)
    assert "subject to market risks" in chunks[0].text


def test_dedupe_identical_blocks(cfg, counter, lex):
    block = fact("Exit load", "Nil")
    chunks, _ = run(make_doc([block, Block(block.section_path, block.kind, block.text)]),
                    cfg, counter, lex)

    assert len(chunks) == 1


def test_oversized_atomic_is_split_visibly(cfg, counter, lex):
    block = Block(["Steps"], "list_item", " ".join(f"step{i}" for i in range(600)))
    chunks, warnings = run(make_doc([block]), cfg, counter, lex)

    assert len(chunks) > 1
    assert all(c.token_count <= cfg.chunking.max_tokens for c in chunks)
    assert not any(c.is_atomic for c in chunks)
    assert any("atomic block over" in w for w in warnings)


def test_ids_are_deterministic(cfg, counter, lex):
    doc = make_doc([fact("Expense ratio", "1.2%"), fact("Exit load", "Nil")])
    first, _ = run(doc, cfg, counter, lex)
    second, _ = run(doc, cfg, counter, lex)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.chunk_id for c in first] == ["test_doc#00", "test_doc#01"]


@pytest.mark.skipif(not list((PROJECT_ROOT / "data/artifacts/01_loaded").glob("*.jsonl")),
                    reason="run --stage load first")
def test_real_corpus_respects_cap(cfg, counter, lex):
    from src.loader import load_documents

    chunks = []
    for doc in load_documents(cfg):
        produced, _ = chunk_document(doc, cfg, counter, lex)
        chunks.extend(produced)

    assert chunks
    assert max(c.token_count for c in chunks) <= cfg.chunking.max_tokens
    assert max(counter.count(c.embed_text) for c in chunks) <= cfg.chunking.max_tokens
    assert lockin_chunk(chunks) is not None

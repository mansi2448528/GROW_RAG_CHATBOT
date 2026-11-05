"""UI/query-service contract: answer, refusal, no-context and PII paths.

Uses an in-memory Chroma collection and a fixed vector so no model weights or
Groq key are needed.
"""

from __future__ import annotations

import csv
import json
import uuid

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")

from config.config import load_config
from services.query import QueryService
from src.lexicon import load_lexicon

DIM = 4
QUERY_VECTOR = (1.0, 0.0, 0.0, 0.0)
URL = {
    "hdfc_elss": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    "hdfc_flexi_cap": "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def isolated_cfg(cfg, tmp_path):
    clone = cfg.model_copy(deep=True)
    clone.app.artifacts_dir = tmp_path / "artifacts"
    clone.artifacts.qa_log = tmp_path / "artifacts" / "qa_log.csv"
    for cached in ("fingerprint", "collection_name", "config_hash"):
        clone.__dict__.pop(cached, None)
    return clone


def vector_for(score: float) -> list[float]:
    return [score, float(np.sqrt(max(0.0, 1.0 - score * score))), 0.0, 0.0]


def make_collection(rows: list[dict]):
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=f"ui_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=[row["chunk_id"] for row in rows],
        embeddings=[row["vector"] for row in rows],
        documents=[row["text"] for row in rows],
        metadatas=[{
            "doc_id": row["doc_id"],
            "source_url": row["source_url"],
            "page_title": "HDFC test page",
            "page_type": row.get("page_type", "groww_scheme"),
            "scheme": row.get("scheme", ""),
            "plan": "direct_growth",
            "amc": "HDFC Mutual Fund",
            "section": "Fund facts",
            "kind": "table_row",
            "fact_keys": ",".join(row.get("fact_keys", [])),
            "is_quarantined": False,
            "fetched_at": "2026-11-01",
            "chunk_strategy": "structure_aware_v1",
            "content_sha256": "abc123",
            "token_count": len(row["text"].split()),
        } for row in rows],
    )
    return collection


class FixedEmbedder:
    def encode(self, texts):
        return np.tile(np.asarray(QUERY_VECTOR, dtype=np.float32),
                       (len(texts), 1)).astype(np.float32)


class NoSearchCollection:
    def query(self, **kwargs):
        raise AssertionError("collection must not be queried")


class NoSearchEmbedder:
    def encode(self, texts):
        raise AssertionError("embedder must not be called")


def row(chunk_id, doc_id, scheme, text, fact_keys, *, score=0.90, url=None):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "scheme": scheme,
        "text": text,
        "fact_keys": fact_keys,
        "vector": vector_for(score),
        "source_url": url or URL[scheme],
    }


def service_for(cfg, tmp_path, collection, embedder=None) -> QueryService:
    small = isolated_cfg(cfg, tmp_path)
    return QueryService(small, collection=collection,
                        embedder=embedder or FixedEmbedder(),
                        lexicon=load_lexicon(small.resolve(small.corpus.lexicon)),
                        use_llm=False)


def test_answer_path_writes_both_artifacts_and_qa_log(cfg, tmp_path):
    collection = make_collection([
        row("flexi_expense", "groww_hdfc_flexi_cap", "hdfc_flexi_cap",
            "Expense ratio: 0.75% per annum", ["expense_ratio"]),
    ])
    service = service_for(cfg, tmp_path, collection)

    card = service.answer("Expense ratio of HDFC Flexi Cap Direct?")

    assert card.status == "ok"
    assert "0.75%" in card.answer
    assert card.source_url == URL["hdfc_flexi_cap"]
    assert card.last_updated == "2026-11-01"
    assert card.trace["validation"]["passed"] is True
    assert card.trace["persisted"] is True

    qid = card.trace["query_id"]
    stage = service.cfg.stage_dir("05_retrieval")
    assert (stage / f"{qid}.json").exists()
    generation = json.loads(
        (service.cfg.stage_dir("06_generation") / f"{qid}.json").read_text("utf-8"))
    assert generation["render_mode"] == "extractive_template"
    assert generation["generation"]["llm_used"] is False
    assert generation["prompt"] is None and generation["llm"] is None
    assert generation["final"]["source_url"] == URL["hdfc_flexi_cap"]
    assert generation["validation"]["citation_valid"] is True

    qa_log = service.cfg.resolve(service.cfg.artifacts.qa_log)
    with qa_log.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and rows[-1]["query_id"] == qid
    assert rows[-1]["status"] == "ok"


def test_advice_refusal_does_not_search_and_has_no_citation(cfg, tmp_path):
    service = service_for(cfg, tmp_path, NoSearchCollection(), NoSearchEmbedder())

    card = service.answer("Should I buy HDFC Mid Cap?")

    assert card.status == "refused"
    assert card.decision == "refuse_advice"
    assert card.source_url is None
    assert "groww.in" not in card.answer

    qid = card.trace["query_id"]
    retrieval = json.loads(
        (service.cfg.stage_dir("05_retrieval") / f"{qid}.json").read_text("utf-8"))
    assert retrieval["search"]["performed"] is False
    generation = json.loads(
        (service.cfg.stage_dir("06_generation") / f"{qid}.json").read_text("utf-8"))
    assert generation["render_mode"] == "refusal_template"
    assert generation["validation"]["scheme_citation"] is False


def test_no_context_path_is_traced_without_inventing_a_source(cfg, tmp_path):
    collection = make_collection([
        row("weak", "groww_hdfc_elss", "hdfc_elss", "Exit load: Nil",
            ["exit_load"], score=0.10),
    ])
    service = service_for(cfg, tmp_path, collection)

    card = service.answer("What is the exit load for HDFC ELSS?")

    assert card.status == "no_context"
    assert card.source_url is None
    assert "couldn't find" in card.answer
    assert card.trace["validation"]["passed"] is True
    qid = card.trace["query_id"]
    generation = json.loads(
        (service.cfg.stage_dir("06_generation") / f"{qid}.json").read_text("utf-8"))
    assert generation["parsed"]["refused"] is True
    assert generation["parsed"]["refusal_reason"] == "not_found"


def test_pii_refusal_writes_nothing(cfg, tmp_path):
    service = service_for(cfg, tmp_path, NoSearchCollection(), NoSearchEmbedder())

    card = service.answer("My PAN is ABCDE1234F, check my folio")

    assert card.status == "refused"
    assert card.decision == "refuse_pii"
    assert "ABCDE1234F" not in card.answer
    assert card.trace["persisted"] is False

    artifacts = service.cfg.resolve(service.cfg.app.artifacts_dir)
    written = [path for path in artifacts.rglob("*") if path.is_file()]
    assert written == []
    for path in written:
        assert "ABCDE1234F" not in path.read_text("utf-8", errors="ignore")


def test_scheme_hint_filters_when_the_question_omits_the_scheme(cfg, tmp_path):
    collection = make_collection([
        row("flexi_exit", "groww_hdfc_flexi_cap", "hdfc_flexi_cap",
            "Exit load: Exit load of 1% if redeemed within 1 year", ["exit_load"]),
        row("elss_exit", "groww_hdfc_elss", "hdfc_elss",
            "Exit load: Nil", ["exit_load"]),
    ])
    service = service_for(cfg, tmp_path, collection)

    card = service.answer("What is the exit load?", scheme_hint="hdfc_elss")

    assert card.status == "ok"
    assert "ELSS" in card.answer
    assert card.source_url == URL["hdfc_elss"]
    qid = card.trace["query_id"]
    trace = json.loads(
        (service.cfg.stage_dir("05_retrieval") / f"{qid}.json").read_text("utf-8"))
    assert trace["analysis"]["scheme"] == "hdfc_elss"
    assert trace["analysis"]["scheme_hint"] == "hdfc_elss"
    assert trace["filters"]["scheme"] == "hdfc_elss"

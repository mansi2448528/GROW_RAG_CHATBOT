"""§6 retrieval invariants: filters, fact match, dedupe, threshold, relax, PII.

No network, no model weights: the collection is an in-memory Chroma index and the
embedder returns a fixed unit vector, so scores are exactly the ones we seed.
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

import numpy as np
import pytest

chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")

from config.config import ConfigError, load_config
from src.artifacts import StageRecorder
from src.retriever import (
    answer_query,
    evaluate_retrieval,
    load_probes,
    make_query_id,
)

DIM = 4
QUERY_VECTOR = (1.0, 0.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def isolated_cfg(cfg, tmp_path):
    clone = cfg.model_copy(deep=True)
    clone.app.artifacts_dir = tmp_path / "artifacts"
    for cached in ("fingerprint", "collection_name", "config_hash"):
        clone.__dict__.pop(cached, None)
    return clone


def vector_for(score: float) -> list[float]:
    """A unit vector whose cosine with QUERY_VECTOR is exactly `score`."""
    score = max(-1.0, min(1.0, score))
    return [score, math.sqrt(max(0.0, 1.0 - score * score)), 0.0, 0.0]


def make_row(chunk_id: str, doc_id: str, text: str, *, fact_keys=(),
             score: float = 0.9, source_url: str | None = None,
             scheme: str = "hdfc_elss", page_type: str = "groww_scheme",
             section: str = "Fund facts") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "fact_keys": list(fact_keys),
        "vector": vector_for(score),
        "source_url": source_url or f"https://groww.in/{doc_id}",
        "scheme": scheme,
        "page_type": page_type,
        "section": section,
    }


def make_collection(rows: list[dict]):
    client = chromadb.EphemeralClient()
    name = f"retrieval_{uuid.uuid4().hex}"
    collection = client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=[row["chunk_id"] for row in rows],
        embeddings=[row["vector"] for row in rows],
        documents=[row["text"] for row in rows],
        metadatas=[{
            "doc_id": row["doc_id"],
            "source_url": row["source_url"],
            "page_title": "HDFC test page",
            "page_type": row["page_type"],
            "scheme": row["scheme"],
            "plan": "direct_growth",
            "amc": "HDFC Mutual Fund",
            "section": row["section"],
            "kind": "definition",
            "fact_keys": ",".join(row["fact_keys"]),
            "is_quarantined": False,
            "fetched_at": "2026-11-01",
            "chunk_strategy": "structure_aware_v1",
            "content_sha256": "abc123",
            "token_count": len(row["text"].split()),
        } for row in rows],
    )
    return collection


class FixedEmbedder:
    def __init__(self, vector=QUERY_VECTOR):
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode(self, texts):
        return np.tile(self.vector, (len(texts), 1)).astype(np.float32)


class NoSearchCollection:
    def query(self, **kwargs):
        raise AssertionError("collection must not be queried")


class NoSearchEmbedder:
    def encode(self, texts):
        raise AssertionError("embedder must not be called")


def test_scheme_filter_fact_match_and_trace(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    collection = make_collection([
        make_row("elss_exit", "groww_hdfc_elss", "Exit load: Nil",
                 fact_keys=["exit_load"], score=0.90,
                 source_url="https://groww.in/elss"),
        make_row("flexi_exit", "groww_hdfc_flexi_cap",
                 "Exit load: Exit load of 1% if redeemed within 1 year",
                 fact_keys=["exit_load"], score=0.90,
                 source_url="https://groww.in/flexi", scheme="hdfc_flexi_cap"),
        make_row("elss_expense", "groww_hdfc_elss", "Expense ratio: 1.2% per annum",
                 fact_keys=["expense_ratio"], score=0.80,
                 source_url="https://groww.in/elss"),
    ])

    plan = answer_query("Exit load for HDFC ELSS?", small, query_id="probe-1",
                        collection=collection, embedder=FixedEmbedder())

    assert plan.decision == "answer"
    assert plan.scheme == "hdfc_elss"
    assert plan.fact_keys == ["exit_load"]
    assert plan.filters == {"is_quarantined": False, "scheme": "hdfc_elss"}
    assert plan.outcome == "hit"
    assert [hit.chunk_id for hit in plan.hits] == ["elss_exit"]

    trace_path = small.stage_dir("05_retrieval") / "probe-1.json"
    trace = json.loads(trace_path.read_text("utf-8"))
    assert trace["decision"] == "answer"
    assert trace["analysis"]["rewrite"] == "exit load of the fund"
    assert trace["filters"] == {"is_quarantined": False, "scheme": "hdfc_elss"}
    assert trace["search"] == {"performed": True, "raw_count": 2, "relaxed": False}
    assert trace["hits"][0]["chunk_id"] == "elss_exit"
    assert trace["hits"][0]["source_url"] == "https://groww.in/elss"
    assert trace["hits"][0]["score"] == pytest.approx(0.90, abs=1e-3)


def test_dedupe_keeps_one_hit_per_source_url(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    same = "https://groww.in/elss"
    collection = make_collection([
        make_row("a", "groww_hdfc_elss", "Minimum SIP: ₹500",
                 fact_keys=["min_sip"], score=0.90, source_url=same),
        make_row("b", "groww_hdfc_elss", "Minimum SIP: ₹500 (copy)",
                 fact_keys=["min_sip"], score=0.89, source_url=same),
        make_row("c", "groww_hdfc_elss", "Minimum SIP: ₹500 (copy 2)",
                 fact_keys=["min_sip"], score=0.88, source_url=same),
        make_row("d", "groww_hdfc_flexi_cap", "Minimum SIP: ₹100",
                 fact_keys=["min_sip"], score=0.87,
                 source_url="https://groww.in/flexi", scheme="hdfc_flexi_cap"),
    ])

    plan = answer_query("What is the minimum SIP?", small, persist=False,
                        collection=collection, embedder=FixedEmbedder())

    assert plan.scheme is None
    assert plan.outcome == "hit"
    assert [hit.chunk_id for hit in plan.hits] == ["a", "d"]
    assert len({hit.source_url for hit in plan.hits}) == len(plan.hits)


def test_below_threshold_is_no_context(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    collection = make_collection([
        make_row("weak", "groww_hdfc_elss", "Exit load: Nil",
                 fact_keys=["exit_load"], score=0.30),
    ])

    plan = answer_query("Exit load for HDFC ELSS?", small, query_id="weak",
                        collection=collection, embedder=FixedEmbedder())

    assert plan.outcome == "no_context"
    assert plan.hits == []
    trace = json.loads(
        (small.stage_dir("05_retrieval") / "weak.json").read_text("utf-8"))
    assert trace["search"] == {"performed": True, "raw_count": 1, "relaxed": False}
    assert trace["hits"] == []


def test_fact_filter_relaxes_once(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    collection = make_collection([
        make_row("unlabelled", "groww_hdfc_elss",
                 "Benchmark: NIFTY 500 Total Return Index",
                 fact_keys=["benchmark"], score=0.90),
    ])

    plan = answer_query("Exit load for HDFC ELSS?", small, query_id="relax",
                        collection=collection, embedder=FixedEmbedder())

    assert plan.outcome == "hit_weak"
    assert [hit.chunk_id for hit in plan.hits] == ["unlabelled"]
    trace = json.loads(
        (small.stage_dir("05_retrieval") / "relax.json").read_text("utf-8"))
    assert trace["search"]["relaxed"] is True


def test_two_named_schemes_do_not_search(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    plan = answer_query("Exit load for HDFC Flexi Cap and HDFC Mid Cap?", small,
                        query_id="ambiguous",
                        collection=NoSearchCollection(),
                        embedder=NoSearchEmbedder())

    assert plan.decision == "answer"
    assert plan.outcome == "no_context"
    assert plan.hits == []
    assert set(plan.filters["ambiguous_schemes"]) == {"hdfc_flexi_cap",
                                                       "hdfc_mid_cap"}
    trace = json.loads(
        (small.stage_dir("05_retrieval") / "ambiguous.json").read_text("utf-8"))
    assert trace["search"]["performed"] is False
    assert trace["filters"]["ambiguous_schemes"]


def test_pii_refusal_is_never_persisted_or_searched(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)

    plan = answer_query("My PAN is ABCDE1234F, check my folio", small,
                        collection=NoSearchCollection(),
                        embedder=NoSearchEmbedder())

    assert plan.decision == "refuse_pii"
    assert plan.outcome == "refused"
    assert plan.hits == []

    stage = small.stage_dir("05_retrieval")
    written = list(stage.rglob("*")) if stage.exists() else []
    assert written == []
    # Belt and braces: nothing anywhere under the artifacts root may contain it.
    artifacts = small.resolve(small.app.artifacts_dir)
    for path in artifacts.rglob("*"):
        if path.is_file():
            assert "ABCDE1234F" not in path.read_text("utf-8", errors="ignore")


def test_non_pii_refusal_may_write_a_trace_but_never_searches(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    query = "Should I buy HDFC Mid Cap?"

    plan = answer_query(query, small, query_id="advice",
                        collection=NoSearchCollection(),
                        embedder=NoSearchEmbedder())

    assert plan.decision == "refuse_advice"
    assert plan.outcome == "refused"
    assert plan.education_url is None
    trace = json.loads(
        (small.stage_dir("05_retrieval") / "advice.json").read_text("utf-8"))
    assert trace["hits"] == []
    assert trace["search"]["performed"] is False


def test_amc_entity_questions_use_lexical_rerank(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    amc_url = "https://groww.in/mutual-funds/amc/hdfc-mutual-funds"
    collection = make_collection([
        make_row("amc_trustees", "groww_hdfc_amc",
                 "Name of trustees: Mr. Vimal Bhandari (Chairman)",
                 score=0.748, source_url=amc_url, scheme="",
                 page_type="groww_amc", section="AMC details"),
        make_row("amc_ceo", "groww_hdfc_amc",
                 "CEO: Navneet Munot(Chief Executive Officer)",
                 score=0.744, source_url=amc_url, scheme="",
                 page_type="groww_amc", section="AMC details"),
    ])

    plan = answer_query("Who is the CEO of HDFC Mutual Fund?", small,
                        query_id="ceo", collection=collection,
                        embedder=FixedEmbedder())

    assert plan.filters == {"is_quarantined": False, "page_type": "groww_amc"}
    assert plan.outcome == "hit"
    assert plan.hits[0].chunk_id == "amc_ceo"
    assert plan.hits[0].lexical == pytest.approx(1.0)


def test_plain_expense_ratio_prefers_the_non_base_row(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    url = "https://groww.in/elss"
    collection = make_collection([
        make_row("base", "groww_hdfc_elss",
                 "Base expense ratio (excluding additional charges): 0.97%",
                 fact_keys=["expense_ratio"], score=0.90, source_url=url),
        make_row("plain", "groww_hdfc_elss", "Expense ratio: 1.2% per annum",
                 fact_keys=["expense_ratio"], score=0.88, source_url=url),
    ])

    plain = answer_query("What is the expense ratio of HDFC ELSS?", small,
                         persist=False, collection=collection,
                         embedder=FixedEmbedder())
    base = answer_query("What is the base expense ratio of HDFC ELSS?", small,
                        persist=False, collection=collection,
                        embedder=FixedEmbedder())

    assert plain.hits[0].chunk_id == "plain"
    assert plain.outcome == "hit"
    assert base.hits[0].chunk_id == "base"


def test_evaluate_retrieval_writes_report_and_traces(cfg, tmp_path):
    small = isolated_cfg(cfg, tmp_path)
    collection = make_collection([
        make_row("elss_exit", "groww_hdfc_elss", "Exit load: Nil",
                 fact_keys=["exit_load"], score=0.90,
                 source_url="https://groww.in/elss"),
        make_row("flexi_sip", "groww_hdfc_flexi_cap", "Minimum SIP: ₹100",
                 fact_keys=["min_sip"], score=0.90,
                 source_url="https://groww.in/flexi", scheme="hdfc_flexi_cap"),
    ])
    probes = tmp_path / "probes.csv"
    probes.write_text(
        "query_id,query,expected_doc_id,expected_fact_key,expected_text\n"
        't1,"Exit load for HDFC ELSS?",groww_hdfc_elss,exit_load,Nil\n'
        't2,"Minimum SIP for HDFC Flexi Cap?",groww_hdfc_flexi_cap,min_sip,₹100\n',
        encoding="utf-8")

    result = evaluate_retrieval(small, recorder=StageRecorder(small),
                                probes_path=probes, collection=collection,
                                embedder=FixedEmbedder())

    assert result.total == 2
    assert result.passed == 2
    assert result.hit_at_3 == 1.0
    assert result.report_path.exists()
    report = result.report_path.read_text("utf-8")
    assert "hit@3: **2/2**" in report
    stage = small.stage_dir("05_retrieval")
    assert (stage / "t1.json").exists() and (stage / "t2.json").exists()
    payload = json.loads((stage / "retrieval_eval.json").read_text("utf-8"))
    assert payload["passed"] == 2 and payload["total"] == 2


def test_load_probes_rejects_missing_columns(tmp_path):
    path = tmp_path / "probes.csv"
    path.write_text("query_id,query\np1,hello\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="missing columns"):
        load_probes(path)


def test_load_probes_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "probes.csv"
    path.write_text(
        "query_id,query,expected_doc_id,expected_fact_key,expected_text\n"
        "p1,Exit load for HDFC ELSS?,groww_hdfc_elss,exit_load,Nil\n"
        "p1,Expense ratio for HDFC ELSS?,groww_hdfc_elss,expense_ratio,1.2%\n",
        encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate query_id"):
        load_probes(path)


def test_query_id_is_stable_and_has_no_raw_text():
    qid = make_query_id("Exit load for HDFC ELSS?")
    assert qid == make_query_id("Exit load for HDFC ELSS?")
    assert qid.startswith("q_") and len(qid) == 14
    assert "ELSS" not in qid and "exit" not in qid.lower()

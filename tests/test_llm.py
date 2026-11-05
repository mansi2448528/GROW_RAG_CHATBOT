"""Optional LLM rephrasing: success, guardrail failures, and honest fallback.

The tests inject a fake phraser, so no Groq key and no network are needed. The
real client is deliberately not exercised here; the UI smoke run covers wiring.
"""

from __future__ import annotations

import json

import pytest

from config.config import load_config
from src.generator import compose_answer, write_generation_trace
from src.lexicon import load_lexicon
from src.llm import LLMResult, build_phraser, fact_preserved
from src.models import QueryPlan, RetrievalHit

URL = "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth"
CLAIM = ("The expense ratio of HDFC Flexi Cap Fund – Direct Plan Growth "
         "is 0.75% per annum.")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def lexicon(cfg):
    return load_lexicon(cfg.resolve(cfg.corpus.lexicon))


class FakePhraser:
    def __init__(self, result: LLMResult):
        self.result = result
        self.calls: list[dict] = []

    def rephrase(self, claim: str, **kwargs):
        self.calls.append({"claim": claim, **kwargs})
        return self.result


def make_plan() -> QueryPlan:
    hit = RetrievalHit(
        chunk_id="groww_hdfc_flexi_cap#01", score=0.9,
        text="Expense ratio: 0.75% per annum",
        section="Fund facts", source_url=URL,
        fact_keys=["expense_ratio"], fetched_at="2026-11-01",
        doc_id="groww_hdfc_flexi_cap",
    )
    return QueryPlan(scheme="hdfc_flexi_cap", fact_keys=["expense_ratio"],
                     hits=[hit], best_score=0.9, outcome="hit")


def result_with(answer: str, *, error: str | None = None) -> LLMResult:
    return LLMResult(
        answer=answer, model="fake-model",
        prompt={"system": "sys", "user": "user",
                "context_chunk_ids": ["groww_hdfc_flexi_cap#01"]},
        text=json.dumps({"answer": answer}),
        parsed={"answer": answer}, latency_ms=12,
        usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        error=error,
    )


def compose(cfg, lexicon, phraser):
    return compose_answer(make_plan(), query="Expense ratio of HDFC Flexi Cap?",
                          cfg=cfg, lexicon=lexicon,
                          corpus_urls={URL}, phraser=phraser)


def test_valid_rewrite_is_used_and_traced(cfg, lexicon):
    phraser = FakePhraser(result_with(
        "The expense ratio for HDFC Flexi Cap Fund – Direct Plan Growth is "
        "0.75% per annum."))

    card = compose(cfg, lexicon, phraser)

    assert card.status == "ok"
    assert card.render_mode == "llm_rephrased"
    assert card.answer == ("The expense ratio for HDFC Flexi Cap Fund – Direct "
                           "Plan Growth is 0.75% per annum.")
    assert card.trace["llm"]["used"] is True
    assert card.trace["llm"]["model"] == "fake-model"
    assert card.trace["llm"]["fact_preserved"] is True
    assert card.trace["validation"]["passed"] is True
    assert phraser.calls[0]["expected_value"] == "0.75% per annum"


@pytest.mark.parametrize("answer,fragment", [
    ("The expense ratio of HDFC Flexi Cap Fund – Direct Plan Growth is "
     "0.85% per annum.", "fact_not_preserved"),
    ("You should buy HDFC Flexi Cap Fund – Direct Plan Growth; its expense "
     "ratio is 0.75% per annum.", "advice_language"),
    ("HDFC Flexi Cap Fund – Direct Plan Growth has an expense ratio of "
     "0.75% per annum, see https://example.com.", "url_in_answer"),
])
def test_unsafe_rewrite_falls_back_to_the_extractive_answer(
        cfg, lexicon, answer, fragment):
    card = compose(cfg, lexicon, FakePhraser(result_with(answer)))

    assert card.status == "ok"
    assert card.render_mode == "extractive_template"
    assert card.answer == CLAIM
    assert card.fallback_used == "llm_fallback_to_extractive"
    assert card.trace["llm"]["used"] is False
    assert fragment in card.trace["llm"]["fallback_reason"]


def test_llm_error_falls_back_without_touching_the_answer(cfg, lexicon):
    phraser = FakePhraser(LLMResult(model="fake-model", error="timeout"))

    card = compose(cfg, lexicon, phraser)

    assert card.status == "ok"
    assert card.answer == CLAIM
    assert card.render_mode == "extractive_template"
    assert card.fallback_used == "llm_error_extractive_fallback"
    assert card.trace["llm"]["error"] == "timeout"


def test_missing_key_disables_the_layer(cfg, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert build_phraser(cfg) is None


def test_trace_records_a_real_llm_attempt(cfg, lexicon, tmp_path):
    clone = cfg.model_copy(deep=True)
    clone.app.artifacts_dir = tmp_path / "artifacts"
    for cached in ("fingerprint", "collection_name", "config_hash"):
        clone.__dict__.pop(cached, None)

    card = compose(cfg, lexicon, FakePhraser(result_with(
        "The expense ratio for HDFC Flexi Cap Fund – Direct Plan Growth is "
        "0.75% per annum.")))
    path = write_generation_trace(clone, make_plan(), card,
                                  query="Expense ratio of HDFC Flexi Cap?",
                                  query_id="qid123")
    payload = json.loads(path.read_text("utf-8"))

    assert payload["provider"] == "groq"
    assert payload["model"] == "fake-model"
    assert payload["prompt"]["system"] == "sys"
    assert payload["llm"]["parsed"]["answer"] == card.answer
    assert payload["generation"]["llm_used"] is True
    assert payload["generation"]["method"] == "llm_rephrased_extractive_verified"
    assert payload["generation"]["extractive_answer"] == CLAIM


def test_fact_preserved_rejects_changed_numbers_and_names():
    assert fact_preserved(CLAIM, CLAIM)
    assert not fact_preserved(CLAIM, CLAIM.replace("0.75%", "0.85%"))
    assert not fact_preserved(
        CLAIM, "The expense ratio of the fund is 0.75% per annum.",
        scheme_name="HDFC Flexi Cap Fund – Direct Plan Growth")

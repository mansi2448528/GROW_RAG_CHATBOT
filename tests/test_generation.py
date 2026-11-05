"""§7 answer composition: extractive templates, refusals, validation.

No model weights are loaded: the generator consumes ``RetrievalHit`` rows exactly
as the retriever would hand them over.
"""

from __future__ import annotations

import pytest

from config.config import load_config
from src.generator import (
    AMBIGUOUS_COPY,
    NO_CONTEXT_COPY,
    compose_answer,
    count_sentences,
    validate_answer,
)
from src.lexicon import load_lexicon
from src.models import AnswerCard, QueryPlan, RetrievalHit

URL = {
    "hdfc_large_cap": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "hdfc_flexi_cap": "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    "hdfc_elss": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    "hdfc_mid_cap": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
    "amc": "https://groww.in/mutual-funds/amc/hdfc-mutual-funds",
}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def lexicon(cfg):
    return load_lexicon(cfg.resolve(cfg.corpus.lexicon))


def hit(scheme: str, text: str, *, fact_keys=(), doc_id: str | None = None,
        score: float = 0.7, url: str | None = None) -> RetrievalHit:
    doc_id = doc_id or f"groww_{scheme}"
    return RetrievalHit(
        chunk_id=f"{doc_id}#01", score=score, text=text, section="Fund facts",
        source_url=url or URL[scheme], fact_keys=list(fact_keys),
        fetched_at="2026-11-01", doc_id=doc_id,
    )


def plan_for(scheme: str | None, fact_keys: list[str],
             hits: list[RetrievalHit], *, filters=None) -> QueryPlan:
    return QueryPlan(scheme=scheme, fact_keys=fact_keys, rewrite="",
                     hits=hits, best_score=hits[0].score if hits else 0.0,
                     outcome="hit" if hits else "no_context",
                     filters=filters or {})


@pytest.mark.parametrize("scheme,fact_key,text,query,expected", [
    ("hdfc_elss", "expense_ratio", "Expense ratio: 1.2% per annum",
     "Expense ratio of HDFC ELSS?",
     "The expense ratio of HDFC ELSS Tax Saver Fund – Direct Plan Growth is 1.2% per annum."),
    ("hdfc_large_cap", "exit_load", "Exit load: Exit load of 1% if redeemed within 1 year",
     "Exit load for HDFC Large Cap?",
     "The exit load for HDFC Large Cap Fund – Direct Plan Growth is 1% if redeemed within 1 year."),
    ("hdfc_mid_cap", "min_sip", "Minimum SIP: ₹100",
     "Minimum SIP for HDFC Mid Cap?",
     "The minimum SIP for HDFC Mid Cap Fund – Direct Plan Growth is ₹100."),
    ("hdfc_elss", "lock_in", "Lock-in: 3 years from the date of investment",
     "ELSS lock-in period?",
     "The lock-in period for HDFC ELSS Tax Saver Fund – Direct Plan Growth is 3 years from the date of investment."),
    ("hdfc_flexi_cap", "riskometer", "Riskometer: Very High Risk",
     "Riskometer of HDFC Flexi Cap?",
     "The riskometer level for HDFC Flexi Cap Fund – Direct Plan Growth is Very High Risk."),
    ("hdfc_flexi_cap", "benchmark",
     "Benchmark: NIFTY 500 Total Return Index (NIFTY 500 TRI)",
     "Benchmark of HDFC Flexi Cap?",
     "The benchmark for HDFC Flexi Cap Fund – Direct Plan Growth is NIFTY 500 Total Return Index (NIFTY 500 TRI)."),
])
def test_extractive_templates(cfg, lexicon, scheme, fact_key, text, query, expected):
    row = hit(scheme, text, fact_keys=[fact_key])
    card = compose_answer(plan_for(scheme, [fact_key], [row]), query=query,
                          cfg=cfg, lexicon=lexicon)

    assert card.status == "ok"
    assert card.decision == "answer"
    assert card.answer == expected
    assert card.source_url == URL[scheme]
    assert card.last_updated == "2026-11-01"
    assert card.render_mode == "extractive_template"
    assert card.trace["validation"]["passed"] is True
    assert count_sentences(card.answer) <= 3


def test_base_expense_ratio_is_labeled_when_asked(cfg, lexicon):
    row = hit("hdfc_elss", "Base expense ratio (excluding additional charges): 0.97%",
              fact_keys=["expense_ratio"])
    card = compose_answer(plan_for("hdfc_elss", ["expense_ratio"], [row]),
                          query="What is the base expense ratio of HDFC ELSS?",
                          cfg=cfg, lexicon=lexicon)

    assert card.status == "ok"
    assert "base expense ratio" in card.answer
    assert "0.97%" in card.answer


def test_plain_expense_ratio_does_not_use_a_base_row(cfg, lexicon):
    row = hit("hdfc_elss", "Base expense ratio (excluding additional charges): 0.97%",
              fact_keys=["expense_ratio"])
    card = compose_answer(plan_for("hdfc_elss", ["expense_ratio"], [row]),
                          query="What is the expense ratio of HDFC ELSS?",
                          cfg=cfg, lexicon=lexicon)

    assert card.status == "no_context"
    assert card.answer == NO_CONTEXT_COPY


@pytest.mark.parametrize("query,text,expected", [
    ("Who is the CEO of HDFC Mutual Fund?",
     "CEO: Navneet Munot(Chief Executive Officer)",
     "The CEO of HDFC Mutual Fund is Navneet Munot."),
    ("What is the total AUM of HDFC Mutual Fund?",
     "Total AUM (as of end of last quarter): ₹9,34,075.66Cr",
     "The total AUM of HDFC Mutual Fund is ₹9,34,075.66Cr, as of the end of the last quarter."),
])
def test_amc_entity_templates(cfg, lexicon, query, text, expected):
    row = hit("amc", text, doc_id="groww_hdfc_amc", url=URL["amc"])
    card = compose_answer(plan_for(None, [], [row]), query=query,
                          cfg=cfg, lexicon=lexicon)

    assert card.status == "ok"
    assert card.answer == expected
    assert card.source_url == URL["amc"]
    assert card.trace["validation"]["passed"] is True


def test_refusal_is_two_sentences_and_has_no_scheme_citation(cfg, lexicon):
    plan = QueryPlan(decision="refuse_advice", outcome="refused",
                     education_url=None)
    card = compose_answer(plan, query="Should I buy HDFC Mid Cap?",
                          cfg=cfg, lexicon=lexicon)

    assert card.status == "refused"
    assert card.source_url is None
    assert count_sentences(card.answer) <= 2
    assert "groww.in" not in card.answer
    assert card.render_mode == "refusal_template"


def test_pii_refusal_never_echoes_the_value(cfg, lexicon):
    plan = QueryPlan(decision="refuse_pii", outcome="refused")
    card = compose_answer(plan, query="My PAN is ABCDE1234F",
                          cfg=cfg, lexicon=lexicon)

    assert card.status == "refused"
    assert "ABCDE1234F" not in card.answer
    assert card.source_url is None


def test_no_context_names_the_covered_scope(cfg, lexicon):
    plan = plan_for(None, ["exit_load"], [])
    card = compose_answer(plan, query="Exit load after 5 years?",
                          cfg=cfg, lexicon=lexicon)

    assert card.status == "no_context"
    assert card.answer == NO_CONTEXT_COPY
    assert "HDFC" in card.answer


def test_exit_load_after_a_longer_period_is_not_answered_with_the_one_year_row(
        cfg, lexicon):
    row = hit("hdfc_flexi_cap",
              "Exit load: Exit load of 1% if redeemed within 1 year",
              fact_keys=["exit_load"])
    card = compose_answer(plan_for("hdfc_flexi_cap", ["exit_load"], [row]),
                          query="What is the exit load after 5 years?",
                          cfg=cfg, lexicon=lexicon)

    assert card.status == "no_context"
    assert card.answer == NO_CONTEXT_COPY
    assert card.source_url is None


def test_exit_load_within_the_disclosed_period_still_answers(cfg, lexicon):
    row = hit("hdfc_flexi_cap",
              "Exit load: Exit load of 1% if redeemed within 1 year",
              fact_keys=["exit_load"])
    card = compose_answer(plan_for("hdfc_flexi_cap", ["exit_load"], [row]),
                          query="What is the exit load if redeemed within 1 year?",
                          cfg=cfg, lexicon=lexicon)

    assert card.status == "ok"
    assert "within 1 year" in card.answer
    assert card.source_url == URL["hdfc_flexi_cap"]


def test_two_named_schemes_get_a_clarification(cfg, lexicon):
    plan = QueryPlan(fact_keys=["exit_load"],
                     filters={"ambiguous_schemes": ["hdfc_flexi_cap", "hdfc_mid_cap"]},
                     outcome="no_context")
    card = compose_answer(plan, query="Exit load of Flexi Cap and Mid Cap?",
                          cfg=cfg, lexicon=lexicon)

    assert card.status == "no_context"
    assert card.answer == AMBIGUOUS_COPY


def test_validation_rejects_advice_language(cfg):
    row = hit("hdfc_elss", "Expense ratio: 1.2% per annum",
              fact_keys=["expense_ratio"])
    plan = plan_for("hdfc_elss", ["expense_ratio"], [row])
    card = AnswerCard(status="ok", answer="You should buy this fund for 1.2%.",
                      source_url=URL["hdfc_elss"], last_updated="2026-11-01")

    result = validate_answer(card, plan, cfg, set(URL.values()),
                             expected_value="1.2%")

    assert result["advice_terms"]
    assert result["passed"] is False


def test_validation_rejects_a_url_not_in_the_retrieved_hits(cfg):
    row = hit("hdfc_elss", "Expense ratio: 1.2% per annum",
              fact_keys=["expense_ratio"])
    plan = plan_for("hdfc_elss", ["expense_ratio"], [row])
    card = AnswerCard(status="ok", answer="The expense ratio is 1.2%.",
                      source_url="https://example.com/fund",
                      last_updated="2026-11-01")

    result = validate_answer(card, plan, cfg, set(URL.values()),
                             expected_value="1.2%")

    assert result["citation_valid"] is False
    assert result["passed"] is False

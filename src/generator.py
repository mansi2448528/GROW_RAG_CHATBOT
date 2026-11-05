"""Extractive-first answer composition and generation tracing (implementation.md §7).

Numeric scheme facts are always rendered from the retrieved row with a
deterministic template. An optional LLM layer (:mod:`src.llm`) may then rephrase
that verified sentence, but only as a cosmetic rewrite:

    extractive template -> validation -> [optional LLM rephrase -> validation]
                        -> answer/no_context

The LLM never supplies a number, name or citation. If it changes a fact, adds
advice/performance language, emits a URL, breaks the sentence limit, or fails at
the network/JSON level, the extractive answer is kept and the fallback is written
to the trace. Without a key the same path runs with no network call and
``llm_used=false``.

Refusals and no-context answers are templates too. Every non-PII path writes
``06_generation/<query_id>.json``; PII refusals write nothing at all.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from config.config import AppConfig
from src.artifacts import StageRecorder
from src.guardrails import detect_pii
from src.lexicon import Lexicon
from src.llm import GroqPhraser, fact_preserved
from src.models import AnswerCard, Decision, QueryPlan, RetrievalHit

GENERATION_STAGE = "06_generation"
DISCLAIMER = (
    "Facts-only. No investment advice. Answers are drawn from official public "
    "pages and are not a recommendation to buy or sell any scheme. Mutual fund "
    "investments are subject to market risks; read all scheme-related documents "
    "carefully."
)

REFUSAL_COPY: dict[Decision, str] = {
    "refuse_advice": (
        "I can only share factual scheme details, so I can't say whether you "
        "should buy, sell, or hold a fund. Please use the scheme's official risk "
        "and cost disclosures when making your own decision."
    ),
    "refuse_performance": (
        "I can only share factual scheme details and can't predict or compare "
        "returns. The scheme's official factsheet is the right place for any "
        "performance information."
    ),
    "refuse_out_of_scope": (
        "I can only answer factual questions about the HDFC mutual fund pages "
        "covered here. This question is outside that scope."
    ),
    "refuse_pii": (
        "Please don't share personal or account information here. I can answer "
        "factual questions about HDFC mutual fund schemes without it."
    ),
}

NO_CONTEXT_COPY = (
    "I couldn't find this in the official pages I cover. I can answer factual "
    "questions about HDFC Large Cap, Flexi Cap, ELSS, Mid Cap, and the HDFC AMC "
    "page."
)
AMBIGUOUS_COPY = (
    "I found more than one HDFC scheme in that question, so I can't choose a "
    "single official source. Please ask about one scheme at a time."
)
EMPTY_COPY = "Please ask a factual question about an HDFC mutual fund scheme or the AMC."

FACT_LABEL_PATTERNS: dict[str, tuple[str, ...]] = {
    "expense_ratio": (r"base expense ratio", r"expense ratio"),
    "exit_load": (r"exit load",),
    "min_sip": (r"minimum sip", r"min\.? sip"),
    "lock_in": (r"lock[- ]?in",),
    "riskometer": (r"riskometer",),
    "benchmark": (r"benchmark",),
    "statement_download": (r"capital gains? statement", r"account statement",
                           r"download statement"),
}

PERFORMANCE_PATTERNS = (
    re.compile(r"\b(?:cagr|xirr|annuali[sz]ed return|past performance)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?%\s*(?:return|returns|p\.a\.|annual)", re.I),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def count_sentences(text: str) -> int:
    """Sentence count that does not split decimals or Indian digit grouping."""
    body = (text or "").strip()
    if not body:
        return 0
    return len([part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", body)
                if part.strip()])


def _clean_value(value: str, label: str) -> str:
    value = re.sub(r"\s*\|\s*", ", ", (value or "").strip())
    value = value.rstrip(" .")
    if label.casefold().startswith("exit load"):
        value = re.sub(r"^exit load(?:\s+of)?\s*", "", value, flags=re.I)
    if label.casefold().startswith("ceo"):
        value = re.sub(r"\s*\([^)]*\)\s*$", "", value)      # drop repeated role
    return value.strip()


def _labelled_value(text: str, patterns: Iterable[str]) -> tuple[str, str] | None:
    """Return ``(label, value)`` for the first ``label: value`` row that matches."""
    compiled = [re.compile(pattern, re.I) for pattern in patterns]
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        label, value = label.strip(), value.strip()
        if value and any(pattern.search(label) for pattern in compiled):
            return label, _clean_value(value, label)
    return None


def _scheme_key(hit: RetrievalHit, plan: QueryPlan,
                lexicon: Lexicon) -> str | None:
    if plan.scheme:
        return plan.scheme
    if hit.doc_id.startswith("groww_"):
        candidate = hit.doc_id[len("groww_"):]
        if candidate in lexicon.scheme_names:
            return candidate
    return None


def _scheme_name(hit: RetrievalHit, plan: QueryPlan, lexicon: Lexicon) -> str:
    key = _scheme_key(hit, plan, lexicon)
    return lexicon.scheme_names.get(key, "the scheme") if key else "the scheme"


def _fact_value(hit: RetrievalHit, fact_key: str, query: str) -> str | None:
    patterns = FACT_LABEL_PATTERNS.get(fact_key)
    if not patterns:
        return None
    found = _labelled_value(hit.text, patterns)
    if not found:
        return None
    label, value = found
    if fact_key == "expense_ratio" and label.casefold().startswith("base"):
        if "base" not in (query or "").casefold():
            return None
    return value


def _render_fact(fact_key: str, value: str, scheme_name: str,
                 query: str) -> str:
    if fact_key == "expense_ratio":
        label = ("base expense ratio" if "base" in (query or "").casefold()
                 else "expense ratio")
        return f"The {label} of {scheme_name} is {value}."
    if fact_key == "exit_load":
        return f"The exit load for {scheme_name} is {value}."
    if fact_key == "min_sip":
        return f"The minimum SIP for {scheme_name} is {value}."
    if fact_key == "lock_in":
        return f"The lock-in period for {scheme_name} is {value}."
    if fact_key == "riskometer":
        return f"The riskometer level for {scheme_name} is {value}."
    if fact_key == "benchmark":
        return f"The benchmark for {scheme_name} is {value}."
    if fact_key == "statement_download":
        return f"The official page describes the statement process as: {value}."
    return f"{fact_key.replace('_', ' ').capitalize()} for {scheme_name} is {value}."


def _amc_answer(query: str, hit: RetrievalHit) -> tuple[str, str] | None:
    """Natural templates for the AMC facts the corpus actually discloses."""
    text = hit.text
    q = (query or "").casefold()

    if re.search(r"\bceo\b|chief executive", q):
        found = _labelled_value(text, (r"^ceo$", r"chief executive"))
        if found:
            return f"The CEO of HDFC Mutual Fund is {found[1]}.", found[1]
    if re.search(r"\bmd\b|managing director", q):
        found = _labelled_value(text, (r"^md$", r"managing director"))
        if found:
            return f"The Managing Director of HDFC Mutual Fund is {found[1]}.", found[1]
    if re.search(r"\baum\b|assets under management", q):
        found = _labelled_value(text, (r"\baum\b", r"assets under management"))
        if found:
            return ("The total AUM of HDFC Mutual Fund is "
                    f"{found[1]}, as of the end of the last quarter."), found[1]
    if re.search(r"trustee", q):
        found = _labelled_value(text, (r"trustee",))
        if found:
            return f"The trustees of HDFC Mutual Fund are {found[1]}.", found[1]
    if re.search(r"sponsor", q):
        found = _labelled_value(text, (r"sponsor",))
        if found:
            return f"The sponsor of HDFC Mutual Fund is {found[1]}.", found[1]
    if re.search(r"incorporat", q):
        found = _labelled_value(text, (r"incorporat",))
        if found:
            return ("HDFC Asset Management Company Ltd. was incorporated on "
                    f"{found[1]}."), found[1]
    if re.search(r"compliance officer", q):
        found = _labelled_value(text, (r"compliance officer",))
        if found:
            return f"The Compliance Officer of HDFC Mutual Fund is {found[1]}.", found[1]
    if re.search(r"\bcio\b|chief investment", q):
        found = _labelled_value(text, (r"^cio$", r"chief investment"))
        if found:
            return (f"The Chief Investment Officer of HDFC Mutual Fund is "
                    f"{found[1]}."), found[1]
    return None


EXIT_LOAD_HORIZON = re.compile(
    r"\b(?:after|beyond|more than)\s+\d+\s*(?:year|month)s?\b", re.I)


def _unsupported_exit_load_horizon(query: str, value: str) -> bool:
    """True when the question asks *after* a period the row does not disclose.

    The corpus states exit load as a condition ("1% if redeemed within 1 year"),
    so "after 5 years" must not be answered with that one-year row. If a page
    ever discloses the asked horizon explicitly, the value will contain it and
    the row is allowed through.
    """
    match = EXIT_LOAD_HORIZON.search(query or "")
    if not match:
        return False
    return match.group(0).casefold() not in (value or "").casefold()


def _hit_answer(plan: QueryPlan, query: str, lexicon: Lexicon
                ) -> tuple[RetrievalHit, str, str | None, str] | None:
    """Return (hit, claim, fact_key, expected_value) for the best rendered row."""
    for fact_key in plan.fact_keys:
        for hit in plan.hits:
            value = _fact_value(hit, fact_key, query)
            if not value:
                continue
            if fact_key == "exit_load" and _unsupported_exit_load_horizon(query, value):
                continue
            claim = _render_fact(fact_key, value,
                                 _scheme_name(hit, plan, lexicon), query)
            return hit, claim, fact_key, value

    for hit in plan.hits:
        amc = _amc_answer(query, hit)
        if amc:
            claim, value = amc
            return hit, claim, None, value
    return None


def _advice_terms(answer: str, cfg: AppConfig) -> list[str]:
    terms: list[str] = []
    for term in cfg.guardrails.advice_terms:
        if re.search(rf"\b{re.escape(term)}\b", answer, re.I):
            terms.append(term)
    return terms


def _performance_terms(answer: str) -> list[str]:
    return [pattern.pattern for pattern in PERFORMANCE_PATTERNS
            if pattern.search(answer)]


def validate_answer(card: AnswerCard, plan: QueryPlan, cfg: AppConfig,
                    corpus_urls: set[str],
                    expected_value: str | None = None) -> dict[str, Any]:
    """V3-V7-style checks. Citation validity is V4."""
    answer = card.answer or ""
    hit_urls = {hit.source_url for hit in plan.hits}
    sentences = count_sentences(answer)
    advice = _advice_terms(answer, cfg)
    performance = _performance_terms(answer)
    pii = detect_pii(answer) is not None
    value_present = bool(expected_value and expected_value in answer)
    urls = re.findall(r"https?://\S+|www\.\S+", answer, re.I)
    citation_valid = bool(card.source_url
                          and card.source_url in hit_urls
                          and card.source_url in corpus_urls)
    max_sentences = cfg.guardrails.max_sentences
    passed = (citation_valid and 0 < sentences <= max_sentences
              and not advice and not performance and not pii and value_present
              and not urls)
    return {
        "citation_valid": citation_valid,
        "sentences": sentences,
        "max_sentences": max_sentences,
        "advice_terms": advice,
        "performance_terms": performance,
        "pii_echo": pii,
        "url_mentions": urls,
        "value_present": value_present,
        "passed": passed,
    }


def _refusal_card(plan: QueryPlan) -> AnswerCard:
    text = REFUSAL_COPY[plan.decision]
    return AnswerCard(
        status="refused",
        answer=text,
        decision=plan.decision,
        education_url=plan.education_url,
        render_mode="refusal_template",
        trace={
            "validation": {
                "sentences": count_sentences(text),
                "max_sentences": 2,
                "scheme_citation": False,
                "education_link_present": bool(plan.education_url),
                "passed": count_sentences(text) <= 2,
            },
            "render_mode": "refusal_template",
            "cited_chunk_id": None,
            "fact_key": None,
            "expected_value": "",
        },
    )


def _no_context_card(plan: QueryPlan) -> AnswerCard:
    if plan.filters.get("ambiguous_schemes"):
        text = AMBIGUOUS_COPY
    elif plan.filters.get("reason") == "empty_query":
        text = EMPTY_COPY
    else:
        text = NO_CONTEXT_COPY
    return AnswerCard(
        status="no_context",
        answer=text,
        decision="answer",
        render_mode="no_context_template",
        trace={
            "validation": {
                "sentences": count_sentences(text),
                "max_sentences": 3,
                "citation_valid": False,
                "passed": count_sentences(text) <= 3,
            },
            "render_mode": "no_context_template",
            "cited_chunk_id": None,
            "fact_key": None,
            "expected_value": "",
        },
    )


def _apply_llm_rephrasing(card: AnswerCard, plan: QueryPlan, *,
                          cfg: AppConfig, phraser: GroqPhraser | None,
                          corpus_urls: set[str], expected: str,
                          scheme_name: str, chunk_id: str) -> None:
    """Optionally rewrite a passing extractive card; never weaken guardrails."""
    if phraser is None:
        card.trace["llm"] = {
            "attempted": False,
            "used": False,
            "reason": "not_configured",
        }
        return

    extractive_answer = card.answer
    result = phraser.rephrase(
        extractive_answer, expected_value=expected,
        scheme_name=scheme_name, chunk_id=chunk_id,
    )
    llm_trace: dict[str, Any] = {
        "attempted": True,
        "used": False,
        "provider": result.provider,
        "model": result.model,
        "prompt": result.prompt,
        "raw_text": result.text,
        "parsed": result.parsed,
        "latency_ms": result.latency_ms,
        "usage": result.usage,
        "error": result.error,
        "extractive_answer": extractive_answer,
    }

    if not result.ok:
        llm_trace["fallback_reason"] = result.error or "empty_answer"
        card.trace["llm"] = llm_trace
        card.fallback_used = "llm_error_extractive_fallback"
        return

    candidate = AnswerCard(
        status="ok",
        answer=result.answer,
        decision="answer",
        source_url=card.source_url,
        last_updated=card.last_updated,
        render_mode="llm_rephrased",
    )
    validation = validate_answer(candidate, plan, cfg, corpus_urls,
                                 expected_value=expected)
    preserved = fact_preserved(extractive_answer, result.answer,
                               expected_value=expected, scheme_name=scheme_name)
    llm_trace["validation"] = validation
    llm_trace["fact_preserved"] = preserved

    if validation["passed"] and preserved:
        llm_trace["used"] = True
        card.answer = result.answer
        card.render_mode = "llm_rephrased"
        card.trace["llm"] = llm_trace
        card.trace["validation"] = validation
        return

    reasons = []
    if not preserved:
        reasons.append("fact_not_preserved")
    for key in ("citation_valid", "value_present", "passed"):
        if key == "passed":
            continue
        if not validation.get(key, True):
            reasons.append(key)
    if validation.get("advice_terms"):
        reasons.append("advice_language")
    if validation.get("performance_terms"):
        reasons.append("performance_language")
    if validation.get("pii_echo"):
        reasons.append("pii_echo")
    if validation.get("url_mentions"):
        reasons.append("url_in_answer")
    if validation.get("sentences", 0) > validation.get("max_sentences", 3):
        reasons.append("sentence_limit")
    llm_trace["fallback_reason"] = ", ".join(dict.fromkeys(reasons)) or "safety_check"
    card.trace["llm"] = llm_trace
    card.fallback_used = "llm_fallback_to_extractive"


def compose_answer(plan: QueryPlan, *, query: str, cfg: AppConfig,
                   lexicon: Lexicon,
                   corpus_urls: set[str] | None = None,
                   phraser: GroqPhraser | None = None) -> AnswerCard:
    """Gate result + retrieved hits -> one answer card.

    The extractive answer is composed and validated first. ``phraser`` is an
    optional cosmetic layer over that already-passing answer.
    """
    if plan.decision == "refuse_pii":
        return _refusal_card(plan)
    if plan.decision != "answer":
        return _refusal_card(plan)
    if not plan.hits:
        return _no_context_card(plan)

    urls = corpus_urls if corpus_urls is not None else {
        row["url"] for row in _read_manifest(cfg)
    }
    rendered = _hit_answer(plan, query, lexicon)
    if not rendered:
        return _no_context_card(plan)

    hit, claim, fact_key, expected = rendered
    card = AnswerCard(
        status="ok",
        answer=claim,
        decision="answer",
        source_url=hit.source_url,
        last_updated=hit.fetched_at,
        render_mode="extractive_template",
        trace={
            "render_mode": "extractive_template",
            "cited_chunk_id": hit.chunk_id,
            "fact_key": fact_key,
            "expected_value": expected,
        },
    )
    validation = validate_answer(card, plan, cfg, urls, expected_value=expected)
    card.trace["validation"] = validation
    if not validation["passed"]:
        fallback = _no_context_card(plan)
        fallback.fallback_used = "extractive_failed_validation"
        fallback.trace["validation"] = validation
        fallback.trace["original_answer"] = claim
        fallback.trace["cited_chunk_id"] = hit.chunk_id
        return fallback

    _apply_llm_rephrasing(
        card, plan, cfg=cfg, phraser=phraser, corpus_urls=urls,
        expected=expected, scheme_name=_scheme_name(hit, plan, lexicon),
        chunk_id=hit.chunk_id,
    )
    return card


def _read_manifest(cfg: AppConfig) -> list[dict[str, str]]:
    from src.loader import read_manifest               # lazy: avoids import cycle

    return read_manifest(cfg)


def write_generation_trace(cfg: AppConfig, plan: QueryPlan, card: AnswerCard,
                           *, query: str, query_id: str,
                           total_latency_ms: int = 0,
                           recorder: StageRecorder | None = None) -> Path:
    """Write ``06_generation/<query_id>.json`` from the card that was rendered.

    The payload mirrors architecture §5.7. ``prompt``/``llm`` are populated only
    when a rephrasing attempt actually happened, so ``llm_used=false`` never
    implies a hidden call. A rejected or failed attempt is recorded verbatim,
    together with the extractive answer that was kept.
    """
    rec = recorder or StageRecorder(cfg)
    trace = card.trace or {}
    validation = trace.get("validation", {})
    llm_trace = trace.get("llm") or {}
    attempted = bool(llm_trace.get("attempted"))
    llm_used = bool(llm_trace.get("used"))
    provider = llm_trace.get("provider") if attempted else "none"
    model = llm_trace.get("model") if attempted else None
    prompt = llm_trace.get("prompt") if attempted else None
    llm_payload = None
    if attempted:
        llm_payload = {
            "text": llm_trace.get("raw_text", ""),
            "parsed": llm_trace.get("parsed"),
            "latency_ms": llm_trace.get("latency_ms", 0),
            "usage": llm_trace.get("usage"),
            "error": llm_trace.get("error"),
        }
    if card.status == "ok":
        parsed = {
            "answer": card.answer,
            "source_url": card.source_url,
            "fact_key": trace.get("fact_key"),
            "confidence": round(plan.best_score, 4),
            "refused": False,
            "refusal_reason": None,
        }
        reason = None
    else:
        reason = (plan.decision if card.status == "refused"
                  else ("ambiguous_schemes"
                        if plan.filters.get("ambiguous_schemes")
                        else "not_found"))
        parsed = {
            "answer": card.answer,
            "source_url": None,
            "fact_key": trace.get("fact_key"),
            "confidence": round(plan.best_score, 4),
            "refused": True,
            "refusal_reason": reason,
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "query_id": query_id,
        "run_id": rec.run_id,
        "created_at": _now(),
        "query": query,
        "decision": plan.decision,
        "outcome": plan.outcome,
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "llm": llm_payload,
        "generation": {
            "method": ("llm_rephrased_extractive_verified" if llm_used
                       else "deterministic_extractive"),
            "llm_used": llm_used,
            "llm_attempted": attempted,
            "llm_fallback_reason": llm_trace.get("fallback_reason"),
            "extractive_answer": llm_trace.get("extractive_answer"),
            "cited_chunk_id": trace.get("cited_chunk_id"),
            "fact_key": trace.get("fact_key"),
        },
        "parsed": parsed,
        "render_mode": card.render_mode,
        "validation": validation,
        "final": {
            "answer": card.answer,
            "source_url": card.source_url,
            "last_updated": card.last_updated,
            "status": card.status,
        },
        "fallback_used": card.fallback_used,
        "total_latency_ms": total_latency_ms,
    }
    return rec.json(GENERATION_STAGE, f"{query_id}.json", payload)

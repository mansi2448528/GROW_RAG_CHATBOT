"""Streamlit UI for the facts-only MF FAQ prototype (implementation.md §7).

The page owns presentation only. Retrieval, answer composition, validation and
artifact writing live in ``services.query`` and ``src``. The trace panel reads
``05_retrieval/<qid>.json`` and ``06_generation/<qid>.json`` directly, so it can
never show more than the pipeline actually recorded.

Run from the project root:

    streamlit run app/ui.py
"""

from __future__ import annotations

import html
import json
import os
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# `streamlit run app/ui.py` puts app/ on sys.path, not the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components

from config.config import AppConfig, ConfigError, load_config
from src.generator import DISCLAIMER
from src.guardrails import detect_pii
from src.loader import read_manifest
from services.query import QueryService

BRAND_LEAD = "HDFC Mutual Fund"
BRAND_ACCENT = "AI Assistant"
BRAND_TITLE = f"{BRAND_LEAD} {BRAND_ACCENT}"
WELCOME_TITLE = "How can I help you today?"
WELCOME = (
    "Ask about HDFC mutual fund schemes and receive a factual answer "
    "backed by an official source and its latest update date."
)
STALE_AFTER_DAYS = 30
SUGGESTIONS = (
    "Expense ratio of HDFC Flexi Cap Direct?",
    "ELSS lock-in period?",
    "Minimum SIP for HDFC Mid Cap?",
    "Exit load for HDFC Large Cap?",
)
SCHEME_OPTIONS: dict[str, str | None] = {
    "Let the assistant decide": None,
    "HDFC Large Cap Fund": "hdfc_large_cap",
    "HDFC Flexi Cap Fund": "hdfc_flexi_cap",
    "HDFC ELSS Tax Saver Fund": "hdfc_elss",
    "HDFC Mid Cap Fund": "hdfc_mid_cap",
    "HDFC Mutual Fund (AMC)": "amc",
}
VALIDATOR_LABELS = {
    "citation_valid": "Citation is a retrieved corpus URL",
    "sentences": "Sentence count",
    "max_sentences": "Sentence limit",
    "advice_terms": "Advice terms",
    "performance_terms": "Performance terms",
    "pii_echo": "PII echo",
    "url_mentions": "URLs inside the answer",
    "value_present": "Asked value appears verbatim",
    "fact_preserved": "Every number and name preserved",
    "scheme_citation": "Scheme citation in refusal",
    "education_link_present": "Education link present",
    "passed": "All blocking checks",
}


# ------------------------------------------------------------------ styling

def _inject_css() -> None:
    st.markdown(
        """
<style>
:root {
  --ink: #0b1f3a;
  --ink-soft: #334155;
  --muted: #64748b;
  --line: #e2e8f0;
  --paper: #ffffff;
  --canvas: #f5f7fb;
  --teal: #0f766e;
  --teal-dark: #115e59;
  --teal-soft: #ecfdf5;
  --amber: #92400e;
  --amber-soft: #fffbeb;
  --blue-soft: #eff6ff;
}

html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
}
.stApp {
  background:
    radial-gradient(circle at 12% 0%, rgba(15,118,110,.10), transparent 30rem),
    radial-gradient(circle at 92% 8%, rgba(59,130,246,.08), transparent 26rem),
    var(--canvas);
}
.block-container {
  max-width: 1080px;
  padding-top: 2.1rem;
  padding-bottom: 5.5rem;
}
[data-testid="stHeader"] { background: transparent; }

.hero {
  position: relative;
  overflow: hidden;
  border-radius: 22px;
  padding: 2.05rem 2.2rem 1.85rem;
  background: linear-gradient(135deg, #0b1f3a 0%, #12365c 58%, #0f4c5c 100%);
  color: #f8fafc;
  box-shadow: 0 18px 42px rgba(11,31,58,.20);
  margin-bottom: 1rem;
}
.hero::after {
  content: "";
  position: absolute;
  width: 24rem; height: 24rem;
  right: -9rem; top: -12rem;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(45,212,191,.30), transparent 66%);
}
.hero .eyebrow {
  font-size: .72rem;
  letter-spacing: .17em;
  text-transform: uppercase;
  font-weight: 700;
  color: #99f6e4;
  margin-bottom: .7rem;
}
.hero h1 {
  font-size: 2.35rem;
  line-height: 1.12;
  letter-spacing: -.025em;
  margin: 0 0 .55rem;
  color: #ffffff;
  font-weight: 750;
}
.hero .sub {
  color: #cbd5e1;
  font-size: .98rem;
  margin: 0 0 .95rem;
}
.hero .lede {
  color: #e2e8f0;
  font-size: 1.04rem;
  line-height: 1.55;
  max-width: 46rem;
  margin: 0;
}
.hero-tags {
  display: flex; flex-wrap: wrap; gap: .5rem;
  margin-top: 1.2rem;
}
.hero-tags span {
  border: 1px solid rgba(148,163,184,.28);
  background: rgba(255,255,255,.07);
  color: #dbeafe;
  border-radius: 999px;
  padding: .28rem .68rem;
  font-size: .74rem;
  letter-spacing: .015em;
}

.disclaimer {
  border: 1px solid #fde68a;
  border-left: 5px solid #f59e0b;
  background: #fffdf5;
  border-radius: 12px;
  padding: .9rem 1.05rem;
  color: #713f12;
  font-size: .87rem;
  line-height: 1.55;
  margin: .8rem 0 1.35rem;
}
.disclaimer strong { color: #78350f; }

.section-label {
  display: flex; align-items: center; gap: .6rem;
  font-size: .74rem;
  letter-spacing: .13em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 750;
  margin: 1.1rem 0 .55rem;
}
.section-label::after {
  content: "";
  flex: 1;
  height: 1px;
  background: var(--line);
}

div[data-testid="stButton"] > button {
  width: 100%;
  min-height: 4.2rem;
  border-radius: 13px;
  border: 1px solid #dbe3ee;
  background: #ffffff;
  color: var(--ink);
  font-weight: 600;
  line-height: 1.35;
  padding: .65rem .85rem;
  box-shadow: 0 3px 10px rgba(15,23,42,.045);
  transition: border-color .15s ease, transform .15s ease, box-shadow .15s ease;
}
div[data-testid="stButton"] > button:hover {
  border-color: #5eead4;
  color: var(--teal-dark);
  transform: translateY(-1px);
  box-shadow: 0 8px 20px rgba(15,118,110,.12);
}
div[data-testid="stButton"] > button:focus:not(:active) {
  border-color: var(--teal);
  color: var(--teal-dark);
}

[data-testid="stSelectbox"] label {
  color: var(--ink-soft);
  font-size: .84rem;
  font-weight: 650;
}

.answer-card {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.25rem 1.4rem 1.1rem;
  box-shadow: 0 10px 28px rgba(15,23,42,.06);
  margin: 1.2rem 0 .7rem;
}
.answer-card.ok { border-top: 4px solid var(--teal); }
.answer-card.warn { border-top: 4px solid #f59e0b; background: #fffefb; }
.answer-card.neutral { border-top: 4px solid #94a3b8; }
.card-top {
  display: flex; align-items: center; justify-content: space-between;
  gap: .8rem; margin-bottom: .8rem;
}
.status-pill {
  display: inline-flex; align-items: center; gap: .38rem;
  border-radius: 999px; padding: .25rem .65rem;
  font-size: .71rem; font-weight: 750; letter-spacing: .055em;
  text-transform: uppercase;
}
.status-pill.ok { color: #065f46; background: var(--teal-soft); }
.status-pill.warn { color: var(--amber); background: var(--amber-soft); }
.status-pill.neutral { color: #475569; background: #f1f5f9; }
.card-kicker { color: var(--muted); font-size: .76rem; }
.answer-text {
  color: var(--ink);
  font-size: 1.09rem;
  line-height: 1.62;
  margin: 0 0 1rem;
  overflow-wrap: anywhere;
}
.source-row {
  border-top: 1px solid #eef2f7;
  padding-top: .8rem;
  font-size: .84rem;
  color: var(--ink-soft);
  overflow-wrap: anywhere;
}
.source-row .label {
  color: var(--muted);
  font-size: .71rem;
  text-transform: uppercase;
  letter-spacing: .09em;
  font-weight: 750;
  margin-right: .45rem;
}
.source-row a { color: var(--teal-dark); text-decoration: none; font-weight: 600; }
.source-row a:hover { text-decoration: underline; }
.freshness {
  margin-top: .4rem;
  color: var(--muted);
  font-size: .79rem;
}
.render-note {
  margin-top: .7rem;
  color: #94a3b8;
  font-size: .72rem;
}
.snapshot-warning {
  margin-top: .65rem;
  border-radius: 8px;
  padding: .45rem .6rem;
  background: var(--amber-soft);
  color: #92400e;
  font-size: .78rem;
}
.refusal-note {
  margin-top: .7rem;
  color: var(--muted);
  font-size: .8rem;
  font-style: italic;
}

[data-testid="stExpander"] {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255,255,255,.72);
  overflow: hidden;
}
[data-testid="stExpander"] summary {
  font-weight: 650;
  color: var(--ink);
}
[data-testid="stChatInput"] {
  border-radius: 14px;
}
.trace-table {
  width: 100%;
  border-collapse: collapse;
  margin: .45rem 0 .8rem;
  font-size: .78rem;
  color: var(--ink-soft);
}
.trace-table th {
  text-align: left;
  background: #f1f5f9;
  color: #475569;
  font-size: .68rem;
  letter-spacing: .055em;
  text-transform: uppercase;
  padding: .45rem .5rem;
  border-bottom: 1px solid var(--line);
}
.trace-table td {
  padding: .45rem .5rem;
  border-bottom: 1px solid #eef2f7;
  vertical-align: top;
  overflow-wrap: anywhere;
}
.trace-table tr:last-child td { border-bottom: 0; }
.trace-table td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.trace-table td.url { max-width: 16rem; }

.footer-note {
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: .78rem;
  line-height: 1.6;
}
</style>
        """,
        unsafe_allow_html=True,
    )
    theme = (PROJECT_ROOT / "app" / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{theme}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers

def _safe(text: Any) -> str:
    return html.escape(str(text or ""))


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            return json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _html_table(rows: list[dict[str, Any]], numeric: set[str] | None = None) -> None:
    """Small HTML table: no pyarrow/NumPy-2 dependency for the trace panel."""
    if not rows:
        return
    numeric = numeric or set()
    columns = list(rows[0])
    head = "".join(f"<th>{_safe(col.replace('_', ' '))}</th>" for col in columns)
    body = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if value is True:
                value = "✓"
            elif value is False:
                value = "✗"
            elif isinstance(value, list):
                value = ", ".join(str(item) for item in value) or "—"
            css = ""
            if col in numeric:
                css = ' class="num"'
            elif col == "source_url":
                css = ' class="url"'
            cells.append(f"<td{css}>{_safe(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f'<table class="trace-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>',
        unsafe_allow_html=True,
    )


def _staleness(last_updated: str | None) -> tuple[int | None, bool]:
    if not last_updated:
        return None, False
    try:
        fetched = date.fromisoformat(last_updated)
    except ValueError:
        return None, False
    days = (date.today() - fetched).days
    return days, days > STALE_AFTER_DAYS


def _render_disclaimer() -> None:
    st.markdown(
        f'<div class="disclaimer"><strong>Facts-only. No investment advice.</strong> '
        f'{_safe(DISCLAIMER.removeprefix("Facts-only. No investment advice.").strip())}'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_topbar(conversation: dict[str, Any]) -> None:
    title = conversation.get("title") or "New chat"
    st.markdown(
        f"""
<div class="chat-topbar">
  <div>
    <div class="topbar-title">{_safe(title)}</div>
    <div class="topbar-sub">HDFC Mutual Fund FAQ · verified official sources</div>
  </div>
  <div class="status-chip"><span class="online-dot"></span> Online</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_welcome(cfg: AppConfig, service: QueryService | None) -> None:
    tags = ["Official-source answers", "No investment advice", "No personal data"]
    if service is not None:
        try:
            tags.append(f"{service.collection.count()} source passages")
        except Exception:
            pass
        if service.llm_enabled:
            tags.append("AI-assisted phrasing")
    st.markdown(
        f"""
<div class="welcome">
  <h1 class="welcome-title">{_safe(BRAND_LEAD)} <span class="welcome-accent">{_safe(BRAND_ACCENT)}</span></h1>
  <p class="welcome-tagline">{_safe(WELCOME_TITLE)}</p>
  <p class="welcome-lede">{_safe(WELCOME)}</p>
  <div class="welcome-tags">{''.join(f'<span>{_safe(tag)}</span>' for tag in tags)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_answer(card) -> None:
    if card.status == "ok":
        days, stale = _staleness(card.last_updated)
        source = (
            f'<div class="source-row"><span class="label">Source</span>'
            f'<a href="{_safe(card.source_url)}" target="_blank" '
            f'rel="noopener noreferrer">{_safe(card.source_url)}</a></div>'
        )
        freshness = (
            '<div class="freshness" title="Date the cited source page was last '
            'updated. Used to assess whether the fact may be stale.">'
            f'Last updated from sources: '
            f'<strong>{_safe(card.last_updated)}</strong></div>'
        )
        warning = ""
        if stale and days is not None:
            warning = (
                '<div class="snapshot-warning">'
                f"The cited source is {days} days old. Please verify this "
                "figure on the official page."
                "</div>"
            )
        llm_trace = (card.trace or {}).get("llm") or {}
        if card.render_mode == "llm_rephrased":
            model = llm_trace.get("model") or "the configured model"
            provider = llm_trace.get("provider") or "the AI provider"
            note = (
                f"Phrased by {provider}/{model} from the verified source entry; "
                "figures and fund names were re-checked before display."
            )
        elif llm_trace.get("attempted"):
            note = (
                "The AI-assisted phrasing did not pass validation, so the "
                "verified source wording is shown instead."
            )
        else:
            note = "Composed directly from the verified source entry."
        body = (
            '<div class="answer-card ok">'
            '<div class="card-top"><span class="status-pill ok">● Verified answer</span>'
            '<span class="card-kicker">Based on the official source</span></div>'
            f'<p class="answer-text">{_safe(card.answer)}</p>'
            f"{source}{freshness}{warning}"
            f'<div class="render-note">{_safe(note)}</div></div>'
        )
    elif card.status == "refused":
        pill = "Request declined"
        if card.education_url:
            link = (
                '<div class="source-row"><span class="label">Learn more</span>'
                f'<a href="{_safe(card.education_url)}" target="_blank" '
                f'rel="noopener noreferrer">{_safe(card.education_url)}</a></div>'
            )
        else:
            link = (
                '<div class="refusal-note">No educational link is available in '
                "the current source set.</div>"
            )
        body = (
            f'<div class="answer-card warn">'
            f'<div class="card-top"><span class="status-pill warn">● {pill}</span>'
            '<span class="card-kicker">Safety check</span></div>'
            f'<p class="answer-text">{_safe(card.answer)}</p>{link}</div>'
        )
    else:
        body = (
            '<div class="answer-card neutral">'
            '<div class="card-top"><span class="status-pill neutral">● Not covered</span>'
            '<span class="card-kicker">Outside the available sources</span></div>'
            f'<p class="answer-text">{_safe(card.answer)}</p>'
            '<div class="refusal-note">No source is cited because the available '
            "pages do not contain a reliable answer.</div></div>"
        )
    st.markdown(body, unsafe_allow_html=True)


def _hit_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rank, hit in enumerate(trace.get("hits", []), start=1):
        rows.append({
            "rank": rank,
            "chunk_id": hit.get("chunk_id", ""),
            "score": hit.get("score", 0),
            "rank_score": hit.get("rank_score", 0),
            "section": hit.get("section", ""),
            "fact_keys": ", ".join(hit.get("fact_keys") or []),
            "source_url": hit.get("source_url", ""),
        })
    return rows


def _raw_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rank, hit in enumerate(trace.get("raw_hits", []), start=1):
        rows.append({
            "rank": rank,
            "chunk_id": hit.get("chunk_id", ""),
            "score": hit.get("score", 0),
            "lexical": hit.get("lexical", 0),
            "section": hit.get("section", ""),
            "source_url": hit.get("source_url", ""),
        })
    return rows


def _render_trace(card, cfg: AppConfig) -> None:
    query_id = (card.trace or {}).get("query_id", "")
    ret_path = cfg.stage_dir("05_retrieval") / f"{query_id}.json"
    gen_path = cfg.stage_dir("06_generation") / f"{query_id}.json"
    retrieval = _load_json(ret_path)
    generation = _load_json(gen_path)

    with st.expander("How this answer was produced", expanded=False):
        st.caption(
            "Rendered directly from the pipeline artifacts — "
            "`05_retrieval/<query_id>.json` and `06_generation/<query_id>.json`. "
            "The UI has no separate logging path."
        )
        if retrieval is None and generation is None:
            if card.decision == "refuse_pii":
                st.info(
                    "This question was refused before retrieval. No trace was "
                    "written, because persisting it would copy the personal data."
                )
            else:
                st.warning("No trace artifacts were found for this query.")
            return

        gate_tab, retrieval_tab, prompt_tab, raw_tab, validators_tab = st.tabs(
            ["Gate", "Retrieval", "Prompt", "Raw output", "Validators"]
        )

        with gate_tab:
            gate = retrieval or {}
            cols = st.columns(4)
            cols[0].metric("Decision", gate.get("decision", card.decision))
            cols[1].metric("Outcome", gate.get("outcome", "—"))
            cols[2].metric("Hits", len(gate.get("hits", [])))
            cols[3].metric("Best score", f"{gate.get('best_score', 0):.3f}")
            analysis = gate.get("analysis") or {}
            st.markdown("**Query analysis**")
            st.json({
                "query_id": query_id,
                "run_id": gate.get("run_id"),
                "scheme": analysis.get("scheme"),
                "scheme_hint": analysis.get("scheme_hint"),
                "fact_keys": analysis.get("fact_keys", []),
                "rewrite": analysis.get("rewrite"),
                "filters": gate.get("filters", {}),
            })
            if gate.get("filters", {}).get("ambiguous_schemes"):
                st.warning("Two or more schemes were named; no single citation is safe.")

        with retrieval_tab:
            trace = retrieval or {}
            search = trace.get("search", {})
            if not search.get("performed"):
                st.info("No vector search was performed for this question.")
            else:
                cfg_block = trace.get("config", {})
                st.caption(
                    f"collection `{cfg_block.get('collection')}` · "
                    f"top_k={cfg_block.get('top_k')} · "
                    f"keep={cfg_block.get('keep')} · "
                    f"threshold={cfg_block.get('score_threshold')} · "
                    f"lexical_boost={cfg_block.get('lexical_boost')}"
                )
                rows = _hit_rows(trace)
                if rows:
                    _html_table(rows, numeric={"rank", "score", "rank_score"})
                else:
                    st.info("No passage passed the score threshold and fact match.")
                with st.expander("Raw top-k before selection", expanded=False):
                    raw = _raw_rows(trace)
                    if raw:
                        _html_table(raw, numeric={"rank", "score", "lexical"})
                    else:
                        st.caption("The raw result set is empty.")

        with prompt_tab:
            if not generation:
                st.info("No generation artifact was written for this question.")
            elif generation.get("prompt") is None:
                st.info(
                    "No prompt was sent. This answer is rendered by a "
                    "deterministic extractive template; `llm_used=false`."
                )
                st.json(generation.get("generation", {}))
            else:
                prompt = generation["prompt"]
                method = generation.get("generation", {}).get("method", "")
                st.caption(
                    f"provider `{generation.get('provider')}` · "
                    f"model `{generation.get('model')}` · method `{method}`"
                )
                st.markdown("**System**")
                st.code(prompt.get("system", ""), language="text")
                st.markdown("**User**")
                st.code(prompt.get("user", ""), language="text")
                st.caption(
                    "Context chunks: "
                    + ", ".join(prompt.get("context_chunk_ids", []))
                )

        with raw_tab:
            if not generation:
                st.info("No generation artifact was written for this question.")
            else:
                st.markdown("**Parsed generation result**")
                st.json(generation.get("parsed", {}))
                llm = generation.get("llm")
                if llm is None:
                    st.caption(
                        "No raw LLM output exists for this answer. "
                        "The extractive renderer wrote the parsed result directly."
                    )
                else:
                    st.markdown("**Raw LLM text**")
                    st.code(llm.get("text", "") or "<empty>", language="json")
                    if llm.get("error"):
                        st.warning(f"LLM error: {llm['error']}")
                    usage = llm.get("usage") or {}
                    if usage:
                        st.caption(
                            "tokens: "
                            f"prompt={usage.get('prompt_tokens', '—')} · "
                            f"completion={usage.get('completion_tokens', '—')} · "
                            f"total={usage.get('total_tokens', '—')}"
                        )
                generation_block = generation.get("generation", {})
                fallback_reason = generation_block.get("llm_fallback_reason")
                if fallback_reason:
                    st.warning(f"Extractive fallback: {fallback_reason}")
                st.caption(
                    f"render_mode: `{generation.get('render_mode')}` · "
                    f"llm_used: `{generation_block.get('llm_used')}` · "
                    f"fallback_used: `{generation.get('fallback_used') or 'none'}` · "
                    f"latency: {generation.get('total_latency_ms', 0)} ms"
                )

        with validators_tab:
            if not generation:
                st.info("No validation record was written for this question.")
            else:
                validation = generation.get("validation", {})
                rows = [
                    {
                        "check": VALIDATOR_LABELS.get(key, key),
                        "value": value,
                    }
                    for key, value in validation.items()
                ]
                _html_table(rows)
                st.caption(
                    "Blocking rule: the answer is downgraded to no-context if a "
                    "citation, sentence limit, advice/performance, PII-echo or "
                    "value-presence check fails."
                )

        st.caption(
            f"Artifacts: `05_retrieval/{ret_path.name}` · "
            f"`06_generation/{gen_path.name}`"
        )


def _render_scope(cfg: AppConfig, *, sidebar: bool = False) -> None:
    ctx = st.sidebar if sidebar else st
    with ctx.expander("Scope, sources and known limits", expanded=False):
        st.markdown(
            "This assistant answers only from the five public Groww pages listed "
            "below. It does not provide investment advice, suitability "
            "assessments, portfolio analysis or return predictions."
        )
        try:
            for row in read_manifest(cfg):
                st.markdown(f"- [{row['id']}]({row['url']})")
        except ConfigError as exc:
            st.warning(str(exc))
        st.markdown(
            "**Privacy:** no PAN, folio, account, email or phone details are "
            "collected. Personal data is detected and refused before retrieval, "
            "and such refusals are never written to artifacts or logs."
        )
        st.markdown(
            "**Answer rendering:** facts are composed from the verified source "
            "entry. When AI-assisted phrasing is enabled, only the wording may "
            "change; every figure and fund name is re-checked, and any mismatch "
            "falls back to the verified source wording."
        )
        st.markdown(
            "**Known limit:** no educational page is included in the current "
            "sources, so a refusal cannot include an education link. The "
            "assistant does not invent one."
        )
    if not sidebar:
        st.markdown(
            '<div class="footer-note">For factual scheme information only. '
            "Mutual fund investments are subject to market risks; read all "
            "scheme-related documents carefully.</div>",
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------- conversations

def _new_conversation() -> dict[str, Any]:
    conversation = {
        "id": uuid.uuid4().hex[:10],
        "title": "New chat",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "messages": [],
    }
    st.session_state.conversations.insert(0, conversation)
    st.session_state.active_id = conversation["id"]
    return conversation


def _init_conversations() -> None:
    st.session_state.setdefault("conversations", [])
    st.session_state.setdefault("active_id", None)
    st.session_state.setdefault("hint_label", "Let the assistant decide")
    st.session_state.setdefault("show_trace", True)
    if not st.session_state.conversations:
        _new_conversation()
    valid_ids = {c["id"] for c in st.session_state.conversations}
    if st.session_state.active_id not in valid_ids:
        st.session_state.active_id = st.session_state.conversations[0]["id"]


def _active_conversation() -> dict[str, Any]:
    _init_conversations()
    for conversation in st.session_state.conversations:
        if conversation["id"] == st.session_state.active_id:
            return conversation
    return _new_conversation()


def _conversation_matches(conversation: dict[str, Any], needle: str) -> bool:
    if not needle:
        return True
    if needle in (conversation.get("title") or "").casefold():
        return True
    return any(
        message.get("role") == "user"
        and needle in str(message.get("content", "")).casefold()
        for message in conversation.get("messages", [])
    )


def _question_title(question: str, card) -> str:
    if card.decision == "refuse_pii":
        return "Personal data question"
    text = " ".join((question or "").split())
    return text[:44] + ("…" if len(text) > 44 else "")


# ---------------------------------------------------------------- sidebar

def _render_sidebar(cfg: AppConfig, service: QueryService | None) -> None:
    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-brand">
  <div class="brand-mark">H</div>
  <div>
    <div class="brand-name">HDFC MF FAQ</div>
    <div class="brand-sub">Verified fund facts</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("New Chat", key="new_chat", type="primary",
                     use_container_width=True):
            if _active_conversation()["messages"]:
                _new_conversation()
            st.rerun()

        search = st.text_input(
            "Search chats",
            placeholder="Search your chats",
            key="chat_search",
        )
        needle = (search or "").strip().casefold()
        st.markdown('<div class="sidebar-label">Chat history</div>',
                    unsafe_allow_html=True)
        visible = [c for c in st.session_state.conversations
                   if _conversation_matches(c, needle)]
        if not visible:
            st.caption("No chats match your search.")
        for conversation in visible:
            active = conversation["id"] == st.session_state.active_id
            count = len(conversation.get("messages", []))
            label = ("▸ " if active else "") + (conversation.get("title")
                                                or "New chat")
            if count:
                label += f"  ·  {count // 2 + count % 2}"
            if st.button(label, key=f"history_{conversation['id']}",
                         use_container_width=True):
                st.session_state.active_id = conversation["id"]
                st.rerun()

        with st.expander("Settings", expanded=False):
            st.selectbox(
                "Optional filter hint",
                list(SCHEME_OPTIONS),
                key="hint_label",
                help="Narrow retrieval when the question does not name a scheme. "
                     "A scheme named in the question always wins.",
            )
            st.checkbox("Show answer trace", key="show_trace",
                        help="Render the artifact-backed 'How this answer was "
                             "produced' panel under each reply.")
            if service is not None:
                mode = ("AI-assisted phrasing · facts re-validated"
                        if service.llm_enabled
                        else "Verified source wording")
                st.caption(f"Answer mode: {mode}")

        _render_scope(cfg, sidebar=True)

        st.markdown(
            """
<div class="sidebar-profile">
  <div class="avatar">G</div>
  <div>
    <div class="profile-name">Guest user</div>
    <div class="profile-sub">Session-only history · not synced</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------- chat actions

def _copy_payload(card) -> str:
    if card.status == "ok" and card.source_url:
        return (f"{card.answer}\n\nSource: {card.source_url}\n"
                f"Last updated from sources: {card.last_updated}")
    return card.answer


def _copy_button(text: str, key: str) -> None:
    """Small clipboard button; works without exposing the answer in page state."""
    payload = json.dumps(text).replace("<", "\\u003c")
    safe_key = "".join(ch if ch.isalnum() else "_" for ch in key)
    components.html(
        f"""
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  .copy-btn {{
    width: 100%; height: 34px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    border: 1px solid #e5e7eb; background: #ffffff; color: #4b5563;
    border-radius: 9px; padding: 0 10px; font-size: 12px; font-weight: 600;
    cursor: pointer; transition: background .12s ease, border-color .12s ease;
  }}
  .copy-btn:hover {{ background: #f9fafb; border-color: #d1d5db; color: #111827; }}
</style>
<button class="copy-btn" id="btn-{safe_key}">Copy</button>
<script>
  const btn = document.getElementById("btn-{safe_key}");
  const value = {payload};
  btn.addEventListener("click", async () => {{
    try {{
      await navigator.clipboard.writeText(value);
    }} catch (err) {{
      const area = document.createElement("textarea");
      area.value = value;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }}
    btn.textContent = "Copied";
    setTimeout(() => {{ btn.textContent = "Copy"; }}, 1400);
  }});
</script>
        """,
        height=38,
        scrolling=False,
    )


def _render_message_actions(card, message: dict[str, Any], *,
                            cfg: AppConfig, service: QueryService) -> None:
    with st.container(key=f"actions-{message['id']}"):
        copy_col, regen_col, _ = st.columns([1.1, 1, 6], gap="small")
        with copy_col:
            _copy_button(_copy_payload(card), f"copy_{message['id']}")
        with regen_col:
            can_regenerate = bool(message.get("query"))
            if st.button("Regenerate", key=f"regen_{message['id']}",
                         disabled=not can_regenerate,
                         use_container_width=True,
                         help="Run the same question through the pipeline again."):
                with st.spinner("Regenerating the answer…"):
                    new_card = service.answer(
                        message["query"],
                        scheme_hint=message.get("hint_value"),
                    )
                message["card"] = new_card
                st.rerun()


# ------------------------------------------------------------------ chat

def _render_suggestions() -> str | None:
    st.markdown('<div class="section-label">Suggested questions</div>',
                unsafe_allow_html=True)
    clicked: str | None = None
    with st.container(key="suggestions"):
        cols = st.columns(2, gap="small")
        for index, suggestion in enumerate(SUGGESTIONS):
            with cols[index % 2]:
                if st.button(suggestion, key=f"suggestion_{index}",
                             use_container_width=True):
                    clicked = suggestion
    st.caption(
        "Factual scheme information only. Investment advice, suitability "
        "assessments and return predictions are not provided."
    )
    return clicked


def _render_messages(cfg: AppConfig, service: QueryService,
                      conversation: dict[str, Any]) -> None:
    for message in conversation["messages"]:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(
                    f'<div class="user-bubble">{_safe(message["content"])}</div>',
                    unsafe_allow_html=True,
                )
            continue

        with st.chat_message("assistant", avatar="🟢"):
            if "error" in message:
                st.error("The query could not be completed.")
                with st.expander("Technical detail", expanded=False):
                    st.code(message["error"], language="text")
                continue
            card = message["card"]
            _render_answer(card)
            _render_message_actions(card, message, cfg=cfg, service=service)
            used_hint = message.get("hint_label")
            if used_hint and used_hint != "Let the assistant decide":
                st.caption(f"Scheme filter applied: **{used_hint}**")
            if st.session_state.show_trace:
                _render_trace(card, cfg)


def _render_chat(cfg: AppConfig, service: QueryService,
                 conversation: dict[str, Any]) -> None:
    _render_topbar(conversation)
    clicked: str | None = None
    if not conversation["messages"]:
        _render_welcome(cfg, service)
        clicked = _render_suggestions()
        _render_disclaimer()
    else:
        _render_disclaimer()
        _render_messages(cfg, service, conversation)

    submitted = st.chat_input("Ask anything about HDFC Mutual Funds...")
    question = clicked or submitted
    if not question:
        return

    hint_label = st.session_state.hint_label
    hint_value = SCHEME_OPTIONS.get(hint_label)
    with st.spinner("Searching the official pages and checking the answer…"):
        try:
            card = service.answer(question, scheme_hint=hint_value)
            shown_question = ("[personal data withheld]"
                              if card.decision == "refuse_pii" else question)
            conversation["messages"].append({
                "id": uuid.uuid4().hex[:10],
                "role": "user",
                "content": shown_question,
            })
            conversation["messages"].append({
                "id": uuid.uuid4().hex[:10],
                "role": "assistant",
                "card": card,
                "query": None if card.decision == "refuse_pii" else question,
                "hint_label": hint_label,
                "hint_value": hint_value,
            })
            if conversation["title"] == "New chat":
                conversation["title"] = _question_title(question, card)
        except Exception as exc:
            stored_question = ("[personal data withheld]"
                               if detect_pii(question) else question)
            conversation["messages"].append({
                "id": uuid.uuid4().hex[:10],
                "role": "user",
                "content": stored_question,
            })
            conversation["messages"].append({
                "id": uuid.uuid4().hex[:10],
                "role": "assistant",
                "error": str(exc),
            })
    st.rerun()


# ------------------------------------------------------------------ page

st.set_page_config(
    page_title=BRAND_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)
_inject_css()


def _apply_streamlit_secrets() -> None:
    """Bridge Streamlit Cloud dashboard secrets into the environment.

    ``config.py`` reads provider keys with ``os.getenv()``. Streamlit Community
    Cloud stores dashboard secrets in ``st.secrets`` instead, so copy the
    top-level string values across without overwriting real environment
    variables. Local runs keep using ``.env`` as before.
    """
    try:
        items = list(st.secrets.items())
    except Exception:                        # no secrets file — local run
        return
    for key, value in items:
        if isinstance(value, str) and key not in os.environ:
            os.environ[key] = value


_apply_streamlit_secrets()


@st.cache_resource(show_spinner="Opening the local retrieval index…")
def _service() -> QueryService:
    return QueryService(load_config())


cfg: AppConfig | None = None
service: QueryService | None = None
startup_error: str | None = None
try:
    cfg = load_config()
    service = _service()
except Exception as exc:                     # fail loud, never answer empty
    startup_error = str(exc)

_init_conversations()

if cfg is not None and service is not None:
    _render_sidebar(cfg, service)
    _render_chat(cfg, service, _active_conversation())
else:
    st.markdown(
        f'<div class="disclaimer"><strong>Facts-only. No investment advice.</strong> '
        f'{_safe(DISCLAIMER.removeprefix("Facts-only. No investment advice.").strip())}'
        "</div>",
        unsafe_allow_html=True,
    )
    st.error("The local retrieval index is not ready, so the assistant is offline.")
    st.code("python -m app.pipeline --stage all", language="powershell")
    if startup_error:
        with st.expander("Technical detail", expanded=False):
            st.code(startup_error, language="text")
    st.stop()

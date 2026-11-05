"""Gate -> analyse -> filter -> search -> dedupe -> threshold (implementation.md §6).

The retriever is deliberately split into three seams so each can be inspected or
tested on its own:

    analyze_query()  lexicon + guardrails -> QueryPlan (no I/O beyond the lexicon)
    retrieve()       one Chroma query + rerank/selection -> SearchOutcome
    answer_query()   the public entry point; owns trace writing and outcome

Nothing here talks to an LLM. A refusal never searches; a PII refusal never even
writes a trace, because the trace would echo the value.

Outputs: ``05_retrieval/<query_id>.json`` (one per query) and, for evaluation,
``05_retrieval/eval_report.md`` + ``retrieval_eval.json``.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config.config import PROJECT_ROOT, AppConfig, ConfigError
from src.artifacts import StageRecorder
from src.embedder import Embedder
from src.guardrails import pick_education_link, pre_gate
from src.lexicon import Lexicon, load_lexicon
from src.models import QueryPlan, RetrievalHit
from src.store import open_collection

TRACE_STAGE = "05_retrieval"
PROBES_PATH = PROJECT_ROOT / "eval" / "probes.csv"
PROBE_COLUMNS = ("query_id", "query", "expected_doc_id", "expected_fact_key",
                 "expected_text")
RETRIEVAL_TARGET = 8

#: Query words that carry no discriminating signal for this corpus. Scheme names
#: are deliberately here: the metadata filter already pins the scheme, and
#: re-adding its name to the rewrite hurts benchmark retrieval.
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how
i if in into is it its me my of on or our please should so tell than that the
their them then there these this those to was we were what when where which who
whom whose why will with would you your
about fund funds scheme schemes plan plans direct growth hdfc mutual india large
cap mid flexi elss tax saver equity
""".split())

#: Entity questions about the AMC itself. Only applied when the query names no
#: scheme and no scheme fact, so it can never steal a scheme fact question.
AMC_INTENT = re.compile(
    r"\b(?:ceo|chief executive|aum|assets under management|trustee|sponsor|"
    r"compliance officer|incorporat(?:ion|ed)|address|amc|asset management company|"
    r"fund house|managing director|md|cio|chief investment|custodian|registrar)\b",
    re.I,
)

_TOKEN = re.compile(r"[a-z0-9]+")
_BASE_EXPENSE = re.compile(r"^\s*base expense", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ analysis

def build_rewrite(query: str, fact_keys: Sequence[str], lexicon: Lexicon) -> str:
    """Natural fact phrases embed better than bare keys or scheme names."""
    phrases = [lexicon.fact_queries.get(key) or key.replace("_", " ")
               for key in fact_keys]
    return " ".join(phrases).strip() or (query or "").strip()


def analyze_query(query: str, cfg: AppConfig,
                  lexicon: Lexicon | None = None,
                  scheme_hint: str | None = None) -> QueryPlan:
    """Apply the pre-gate, then derive scheme/facts/rewrite/filters.

    ``scheme_hint`` is the UI's optional selector (a scheme key or ``"amc"``).
    It only supplies context the question itself omits; a scheme named in the
    question always wins, and an ambiguous question stays ambiguous.
    """
    text = (query or "").strip()
    lex = lexicon or load_lexicon(cfg.resolve(cfg.corpus.lexicon))
    decision = pre_gate(text, cfg, lex)

    if decision == "refuse_pii":
        # No education lookup, no retrieval, no persistence: the query contains
        # a value that must never be copied into an artifact.
        return QueryPlan(decision=decision, outcome="refused")

    if decision != "answer":
        return QueryPlan(decision=decision, outcome="refused",
                         education_url=pick_education_link(cfg))

    if not text:
        return QueryPlan(filters={"reason": "empty_query"}, outcome="no_context")

    schemes = lex.schemes(text)
    fact_keys = lex.fact_keys(text)
    rewrite = build_rewrite(text, fact_keys, lex)

    if len(schemes) > 1:
        # "Flexi and Mid Cap exit load" has no single correct citation. Search
        # one scheme and the answer silently becomes wrong for the other.
        return QueryPlan(fact_keys=fact_keys, rewrite=rewrite,
                         filters={"ambiguous_schemes": schemes},
                         outcome="no_context")

    scheme = schemes[0] if schemes else None
    hint = (scheme_hint or "").strip() or None
    if hint == "amc":
        hint_scheme = None
    elif hint in lex.scheme_patterns:
        hint_scheme = hint
    else:
        hint_scheme = None
        hint = None

    filters: dict[str, Any] = {}
    if cfg.retrieval.exclude_quarantined:
        filters["is_quarantined"] = False
    if scheme is None and hint_scheme:
        scheme = hint_scheme
    if cfg.retrieval.filter_by_scheme and scheme:
        filters["scheme"] = scheme
    if hint == "amc" and not scheme:
        filters["page_type"] = "groww_amc"
    elif AMC_INTENT.search(text) and not scheme and not fact_keys:
        filters["page_type"] = "groww_amc"

    return QueryPlan(scheme=scheme, fact_keys=fact_keys, rewrite=rewrite,
                     filters=filters)


def _needs_search(plan: QueryPlan) -> bool:
    return (plan.decision == "answer"
            and "ambiguous_schemes" not in plan.filters
            and "reason" not in plan.filters)


# ------------------------------------------------------------------ rerank

def query_terms(text: str) -> set[str]:
    return {token for token in _TOKEN.findall((text or "").lower())
            if len(token) > 1 and token not in STOPWORDS}


def lexical_overlap(query: str, text: str) -> float:
    """Share of informative query terms present in the chunk, 0..1."""
    terms = query_terms(query)
    if not terms:
        return 0.0
    return len(terms & query_terms(text)) / len(terms)


def _is_base_expense(text: str) -> bool:
    return bool(_BASE_EXPENSE.match(text or ""))


def _is_shadowed(hit: RetrievalHit, fact_keys: Sequence[str], query: str) -> bool:
    """Skip the 'base expense ratio' row for a plain expense-ratio question."""
    return ("expense_ratio" in fact_keys
            and "base" not in (query or "").lower()
            and _is_base_expense(hit.text))


def _chroma_where(filters: dict[str, Any]) -> dict | None:
    """Chroma 1.5.9 needs an explicit $and for more than one key."""
    clauses = [{key: value} for key, value in filters.items()
               if isinstance(value, (str, bool, int, float))]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _hits_from(result: dict, query: str, boost: float) -> list[RetrievalHit]:
    rows = zip(result.get("ids", [[]])[0], result.get("documents", [[]])[0],
               result.get("metadatas", [[]])[0], result.get("distances", [[]])[0])
    hits: list[RetrievalHit] = []
    for chunk_id, document, metadata, distance in rows:
        metadata = metadata or {}
        score = 1.0 - float(distance)          # cosine distance -> similarity
        lexical = lexical_overlap(query, document or "")
        hits.append(RetrievalHit(
            chunk_id=chunk_id,
            score=score,
            text=document or "",
            section=metadata.get("section", ""),
            source_url=metadata.get("source_url", ""),
            fact_keys=[k for k in (metadata.get("fact_keys") or "").split(",") if k],
            fetched_at=metadata.get("fetched_at", ""),
            doc_id=metadata.get("doc_id", ""),
            rank_score=score + boost * lexical,
            lexical=lexical,
        ))
    return hits


def _pick(hits: Sequence[RetrievalHit], cfg: AppConfig, fact_keys: Sequence[str],
          query: str, *, require_fact: bool) -> list[RetrievalHit]:
    kept: list[RetrievalHit] = []
    seen: set[str] = set()
    for hit in hits:                                    # already rank-sorted
        if hit.score < cfg.retrieval.score_threshold:
            continue
        if hit.source_url in seen:                      # one citation per page
            continue
        if require_fact and fact_keys and not (set(hit.fact_keys) & set(fact_keys)):
            continue
        if _is_shadowed(hit, fact_keys, query):
            continue
        kept.append(hit)
        seen.add(hit.source_url)
        if len(kept) == cfg.retrieval.keep:
            break
    return kept


def _select(hits: Sequence[RetrievalHit], cfg: AppConfig,
            fact_keys: Sequence[str], query: str) -> tuple[list[RetrievalHit], bool]:
    strict = _pick(hits, cfg, fact_keys, query, require_fact=True)
    if strict:
        return strict, False
    if fact_keys:
        # One honest relaxation: the chunk may be right even when its fact_key
        # label is missing. Never relax the score threshold or the scheme filter.
        relaxed = _pick(hits, cfg, fact_keys, query, require_fact=False)
        if relaxed:
            return relaxed, True
    return [], False


@dataclass
class SearchOutcome:
    hits: list[RetrievalHit] = field(default_factory=list)
    raw: list[RetrievalHit] = field(default_factory=list)
    relaxed: bool = False


def retrieve(plan: QueryPlan, cfg: AppConfig, *, query: str, collection,
             embedder) -> SearchOutcome:
    """Run exactly one filtered vector query, then rerank and select."""
    if not _needs_search(plan):
        return SearchOutcome()

    vector = embedder.encode([plan.rewrite])[0]
    raw = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=cfg.retrieval.top_k,
        where=_chroma_where(plan.filters),
        include=["documents", "metadatas", "distances"],
    )
    hits = _hits_from(raw, query, cfg.retrieval.lexical_boost)
    hits.sort(key=lambda hit: (-hit.rank_score, -hit.score, hit.chunk_id))
    kept, relaxed = _select(hits, cfg, plan.fact_keys, query)
    return SearchOutcome(hits=kept, raw=hits, relaxed=relaxed)


# ------------------------------------------------------------------ trace

def make_query_id(query: str) -> str:
    digest = hashlib.sha256((query or "").strip().encode("utf-8")).hexdigest()[:12]
    return f"q_{digest}"


def _hit_dict(hit: RetrievalHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "score": round(hit.score, 4),
        "rank_score": round(hit.rank_score, 4),
        "lexical": round(hit.lexical, 3),
        "section": hit.section,
        "source_url": hit.source_url,
        "fact_keys": hit.fact_keys,
        "fetched_at": hit.fetched_at,
        "text": hit.text,
    }


def write_trace(cfg: AppConfig, plan: QueryPlan, *, query: str, query_id: str,
                recorder: StageRecorder | None = None,
                search: SearchOutcome | None = None,
                scheme_hint: str | None = None) -> Path:
    """Persist the stage-by-stage retrieval decision for one question."""
    rec = recorder or StageRecorder(cfg)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "query_id": query_id,
        "run_id": rec.run_id,
        "created_at": _now(),
        "query": query,
        "decision": plan.decision,
        "outcome": plan.outcome,
        "analysis": {
            "scheme": plan.scheme,
            "fact_keys": plan.fact_keys,
            "rewrite": plan.rewrite,
            "scheme_hint": scheme_hint,
        },
        "filters": plan.filters,
        "config": {
            "collection": cfg.collection_name,
            "config_hash": cfg.config_hash,
            "embedding_model": cfg.embedding.model_id,
            "embedding_fingerprint": cfg.fingerprint,
            "top_k": cfg.retrieval.top_k,
            "keep": cfg.retrieval.keep,
            "score_threshold": cfg.retrieval.score_threshold,
            "lexical_boost": cfg.retrieval.lexical_boost,
        },
        "search": {
            "performed": search is not None,
            "raw_count": len(search.raw) if search else 0,
            "relaxed": search.relaxed if search else False,
        },
        "best_score": round(plan.best_score, 4),
        "hits": [_hit_dict(hit) for hit in plan.hits],
        "raw_hits": [_hit_dict(hit) for hit in (search.raw if search else [])],
        "education_url": plan.education_url,
    }
    return rec.json(TRACE_STAGE, f"{query_id}.json", payload)


# ------------------------------------------------------------------ entry point

def answer_query(query: str, cfg: AppConfig, *, recorder: StageRecorder | None = None,
                 query_id: str | None = None, collection=None, embedder=None,
                 lexicon: Lexicon | None = None,
                 persist: bool = True,
                 scheme_hint: str | None = None) -> QueryPlan:
    """Full §6 path for one question. No LLM, no network."""
    text = (query or "").strip()
    lex = lexicon or load_lexicon(cfg.resolve(cfg.corpus.lexicon))
    plan = analyze_query(text, cfg, lex, scheme_hint=scheme_hint)
    qid = query_id or make_query_id(text)

    if plan.decision == "refuse_pii":
        return plan                                  # never write, never search

    search: SearchOutcome | None = None
    if _needs_search(plan):
        if collection is None:
            collection = open_collection(cfg)
        if embedder is None:
            embedder = Embedder(cfg)
        search = retrieve(plan, cfg, query=text, collection=collection,
                          embedder=embedder)
        plan.hits = search.hits
        plan.best_score = max((hit.score for hit in search.hits), default=0.0)
        if search.hits:
            plan.outcome = "hit_weak" if search.relaxed else "hit"
        else:
            plan.outcome = "no_context"

    if persist and cfg.artifacts.write_per_query:
        write_trace(cfg, plan, query=text, query_id=qid, recorder=recorder,
                    search=search, scheme_hint=scheme_hint)
    return plan


# ------------------------------------------------------------------ evaluation

def load_probes(path: str | Path | None = None,
                cfg: AppConfig | None = None) -> list[dict[str, str]]:
    """Read ``eval/probes.csv`` and validate its columns before any query runs."""
    probe_path = Path(path) if path else (
        cfg.resolve(Path("eval/probes.csv")) if cfg else PROBES_PATH)
    if not probe_path.exists():
        raise ConfigError(f"probes not found: {probe_path}")
    with probe_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        missing = [column for column in PROBE_COLUMNS if column not in columns]
        if missing:
            raise ConfigError(f"{probe_path}: missing columns: {', '.join(missing)}")
        probes: list[dict[str, str]] = []
        for line, row in enumerate(reader, start=2):
            query = (row.get("query") or "").strip()
            query_id = (row.get("query_id") or "").strip()
            if not query:
                raise ConfigError(f"{probe_path}:{line}: empty query")
            if not query_id:
                raise ConfigError(f"{probe_path}:{line}: empty query_id")
            probes.append({column: (row.get(column) or "").strip()
                           for column in PROBE_COLUMNS})
    if not probes:
        raise ConfigError(f"{probe_path}: no probes")
    ids = [probe["query_id"] for probe in probes]
    duplicates = sorted({query_id for query_id in ids if ids.count(query_id) > 1})
    if duplicates:
        raise ConfigError(f"{probe_path}: duplicate query_id(s): "
                          f"{', '.join(duplicates)}")
    return probes


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def _probe_match(plan: QueryPlan,
                 probe: dict[str, str]) -> tuple[int | None, RetrievalHit | None]:
    """A probe hits when the expected chunk is in the top `keep` results."""
    for rank, hit in enumerate(plan.hits, start=1):
        if probe["expected_doc_id"] and hit.doc_id != probe["expected_doc_id"]:
            continue
        if (probe["expected_fact_key"]
                and probe["expected_fact_key"] not in hit.fact_keys):
            continue
        if (probe["expected_text"]
                and _normalise(probe["expected_text"]) not in _normalise(hit.text)):
            continue
        return rank, hit
    return None, None


@dataclass
class RetrievalEval:
    rows: list[dict[str, Any]]
    passed: int
    total: int
    target: int = RETRIEVAL_TARGET
    report_path: Path | None = None

    @property
    def hit_at_3(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def ok(self) -> bool:
        return self.passed >= self.target


def _eval_report(result: RetrievalEval, run_id: str) -> str:
    lines = [
        "# Retrieval evaluation",
        "",
        f"- probes: **{result.total}**",
        f"- hit@3: **{result.passed}/{result.total}** "
        f"({result.hit_at_3:.0%}) · target ≥ {result.target}",
        f"- run_id: `{run_id}`",
        "",
        "| query_id | query | expected | predicted | section | rank | score | outcome | |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result.rows:
        lines.append(
            f"| `{row['query_id']}` | {row['query']} | {row['expected']} "
            f"| {row['predicted']} | {row['section'] or '—'} | {row['rank'] or '—'} "
            f"| {row['score']:.3f} | `{row['outcome']}` "
            f"| {'✓' if row['ok'] else '✗'} |")
    misses = [row for row in result.rows if not row["ok"]]
    if misses:
        lines += ["", "## Misses", ""]
        lines += [f"- `{row['query_id']}`: expected {row['expected']}, "
                  f"got {row['predicted']} ({row['outcome']})" for row in misses]
    else:
        lines += ["", "All probes hit within the top 3.", ""]
    return "\n".join(lines)


def evaluate_retrieval(cfg: AppConfig, *, recorder: StageRecorder | None = None,
                       probes_path: str | Path | None = None, collection=None,
                       embedder=None) -> RetrievalEval:
    """Run every probe, write one trace per probe plus a summary report."""
    rec = recorder or StageRecorder(cfg)
    probes = load_probes(probes_path, cfg)
    lexicon = load_lexicon(cfg.resolve(cfg.corpus.lexicon))
    if collection is None:
        collection = open_collection(cfg)
    if embedder is None:
        embedder = Embedder(cfg)

    rows: list[dict[str, Any]] = []
    for probe in probes:
        plan = answer_query(probe["query"], cfg, recorder=rec,
                            query_id=probe["query_id"], collection=collection,
                            embedder=embedder, lexicon=lexicon, persist=True)
        rank, hit = _probe_match(plan, probe)
        top = plan.hits[0] if plan.hits else None
        shown = hit or top
        expected = probe["expected_doc_id"] + (
            f" · {probe['expected_fact_key']}" if probe["expected_fact_key"] else "")
        rows.append({
            "query_id": probe["query_id"],
            "query": probe["query"],
            "expected": expected,
            "predicted": (f"{shown.doc_id} · "
                          f"{','.join(shown.fact_keys) or '—'}") if shown else "—",
            "predicted_chunk": shown.chunk_id if shown else "",
            "section": shown.section if shown else "",
            "rank": rank or 0,
            "score": shown.score if shown else 0.0,
            "outcome": plan.outcome,
            "decision": plan.decision,
            "ok": rank is not None,
        })

    passed = sum(1 for row in rows if row["ok"])
    result = RetrievalEval(rows=rows, passed=passed, total=len(rows))
    result.report_path = rec.markdown(TRACE_STAGE, "eval_report.md",
                                      _eval_report(result, rec.run_id))
    rec.json(TRACE_STAGE, "retrieval_eval.json", {
        "run_id": rec.run_id,
        "probes": str(probes_path or PROBES_PATH),
        "total": result.total,
        "passed": result.passed,
        "hit_at_3": round(result.hit_at_3, 4),
        "target": result.target,
        "ok": result.ok,
        "rows": [{**row, "score": round(row["score"], 4)} for row in rows],
        "created_at": _now(),
    })
    rec.manifest(TRACE_STAGE, {
        "probes": result.total,
        "passed": result.passed,
        "hit_at_3": round(result.hit_at_3, 4),
        "target": result.target,
    })
    return result

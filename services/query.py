"""Single UI/API request path: gate -> retrieve -> compose -> validate -> card.

The Streamlit page must not contain retrieval or answer logic. It calls this
service, then renders the two artifacts the pipeline already wrote:

    05_retrieval/<query_id>.json   (written by src.retriever)
    06_generation/<query_id>.json  (written here, from src.generator)

That keeps the demo honest: the trace panel has exactly one source of truth.

No LLM key is required. Answers are always composed and validated as
deterministic extractive templates; when ``GROQ_API_KEY`` and ``LLM_MODEL`` are
present, an optional phraser may rewrite the verified sentence. The generation
trace records the attempt, the raw output, and any fallback reason honestly.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from config.config import AppConfig
from src.artifacts import StageRecorder
from src.embedder import Embedder
from src.generator import compose_answer, write_generation_trace
from src.lexicon import Lexicon, load_lexicon
from src.llm import build_phraser
from src.loader import read_manifest
from src.models import AnswerCard
from src.retriever import answer_query, make_query_id
from src.store import open_collection


class QueryService:
    """Reuse the collection/embedding model across Streamlit reruns."""

    def __init__(self, cfg: AppConfig, *, collection=None, embedder=None,
                 lexicon: Lexicon | None = None,
                 recorder: StageRecorder | None = None,
                 phraser=None, use_llm: bool = True):
        self.cfg = cfg
        self.recorder = recorder or StageRecorder(cfg)
        self.lexicon = lexicon or load_lexicon(cfg.resolve(cfg.corpus.lexicon))
        self.collection = collection if collection is not None else open_collection(cfg)
        self.embedder = embedder if embedder is not None else Embedder(cfg)
        self.corpus_urls = {row["url"] for row in read_manifest(cfg)}
        if phraser is None and use_llm:
            phraser = build_phraser(cfg)
        self.phraser = phraser

    @property
    def llm_enabled(self) -> bool:
        return self.phraser is not None

    def answer(self, query: str, *, scheme_hint: str | None = None) -> AnswerCard:
        """Answer one question and persist the generation trace (except PII)."""
        started = time.monotonic()
        plan = answer_query(
            query, self.cfg, recorder=self.recorder, collection=self.collection,
            embedder=self.embedder, lexicon=self.lexicon, scheme_hint=scheme_hint,
        )
        query_id = make_query_id((query or "").strip())
        card = compose_answer(plan, query=query, cfg=self.cfg,
                              lexicon=self.lexicon,
                              corpus_urls=self.corpus_urls,
                              phraser=self.phraser)
        card.trace["query_id"] = query_id

        if plan.decision == "refuse_pii":
            # No trace, no qa_log: the question must leave no persisted copy.
            card.trace["persisted"] = False
            return card

        latency_ms = int((time.monotonic() - started) * 1000)
        if self.cfg.artifacts.write_per_query:
            trace_path = write_generation_trace(
                self.cfg, plan, card, query=query, query_id=query_id,
                total_latency_ms=latency_ms, recorder=self.recorder,
            )
            card.trace["generation_trace"] = str(trace_path)
            card.trace["retrieval_trace"] = str(
                self.cfg.stage_dir("05_retrieval") / f"{query_id}.json")
            self._append_qa_log(plan, card, query, query_id, latency_ms)
        card.trace["persisted"] = bool(self.cfg.artifacts.write_per_query)
        return card

    def _append_qa_log(self, plan, card: AnswerCard, query: str, query_id: str,
                       latency_ms: int) -> Path:
        path = self.cfg.resolve(self.cfg.artifacts.qa_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["query_id", "redacted_query", "decision", "fact_key",
                  "source_url", "status", "latency_ms"]
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({
                "query_id": query_id,
                "redacted_query": query,        # non-PII path only
                "decision": card.decision,
                "fact_key": card.trace.get("fact_key") or "",
                "source_url": card.source_url or "",
                "status": card.status,
                "latency_ms": latency_ms,
            })
        return path

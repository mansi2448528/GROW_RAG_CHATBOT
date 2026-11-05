"""Artifact-level data contracts (implementation.md §2.4, §3, §5, §6).

These dataclasses are the hand-off format between stages: every stage reads the
previous stage's records off disk and writes its own. Changing a field here is a
schema change and must be reflected in the artifact writers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal, Optional

PageType = Literal["groww_scheme", "groww_amc", "amc_page", "sebi_page", "amfi_page"]
Decision = Literal["answer", "refuse_advice", "refuse_pii", "refuse_performance", "refuse_out_of_scope"]
Outcome = Literal["hit", "hit_weak", "no_context", "refused"]


def _default_date() -> date:
    return date.today()


@dataclass
class DocMeta:
    doc_id: str
    source_url: str
    page_title: str
    page_type: PageType
    scheme: Optional[str]          # None for AMC / regulator pages
    amc: str
    plan: str                      # direct_growth | na
    fetched_at: date = field(default_factory=_default_date)
    education: bool = False


@dataclass
class Block:
    """One structural unit of a page. `kind` drives chunk atomicity (§3.1)."""
    section_path: list[str]
    kind: str                      # table_row | definition | list_item | paragraph | heading
    text: str
    fact_key: Optional[str] = None  # set when the block is a known scheme fact
    rows: Optional[list[list[str]]] = None


@dataclass
class Document:
    meta: DocMeta
    blocks: list[Block]
    content_sha256: str
    http_status: int = 200
    parser: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Document":
        meta = DocMeta(**{**raw["meta"], "fetched_at": date.fromisoformat(raw["meta"]["fetched_at"])})
        blocks = [Block(**b) for b in raw["blocks"]]
        return cls(meta=meta, blocks=blocks, content_sha256=raw["content_sha256"],
                   http_status=raw.get("http_status", 200),
                   parser=raw.get("parser", ""), warnings=raw.get("warnings", []))

    @classmethod
    def from_jsonl(cls, line: str) -> "Document":
        return cls.from_dict(json.loads(line))


@dataclass
class Chunk:
    chunk_id: str                  # f"{doc_id}#NN" — deterministic
    doc_id: str
    text: str                      # answerable / display text
    embed_text: str                # title + section + text (§4)
    section_path: list[str]
    kind: str
    fact_keys: list[str]           # vocabulary from config/lexicon.yaml
    is_atomic: bool
    is_quarantined: bool
    token_count: int
    meta: DocMeta
    chunk_strategy: str = "structure_aware_v1"
    content_sha256: str = ""
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Chunk":
        payload = dict(raw)
        meta = payload.pop("meta")
        meta = DocMeta(**{**meta, "fetched_at": date.fromisoformat(meta["fetched_at"])})
        return cls(meta=meta, **payload)

    @classmethod
    def from_jsonl(cls, line: str) -> "Chunk":
        return cls.from_dict(json.loads(line))


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float                   # cosine similarity (threshold is applied here)
    text: str
    section: str
    source_url: str
    fact_keys: list[str]
    fetched_at: str
    doc_id: str = ""
    rank_score: float = 0.0        # cosine + small lexical bonus (§6 rerank)
    lexical: float = 0.0           # informative-query-term overlap, 0..1


@dataclass
class QueryPlan:
    decision: Decision = "answer"
    scheme: Optional[str] = None
    fact_keys: list[str] = field(default_factory=list)
    rewrite: str = ""
    filters: dict[str, Any] = field(default_factory=dict)
    hits: list[RetrievalHit] = field(default_factory=list)
    best_score: float = 0.0
    outcome: Outcome = "no_context"
    education_url: Optional[str] = None


@dataclass
class AnswerCard:
    status: Literal["ok", "refused", "no_context"]
    answer: str
    decision: Decision = "answer"
    source_url: Optional[str] = None
    last_updated: Optional[str] = None
    education_url: Optional[str] = None
    render_mode: str = "extractive_template"
    fallback_used: Optional[str] = None
    trace: dict[str, Any] = field(default_factory=dict)

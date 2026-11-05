"""PII/advice pre-gate and refusal helpers (implementation.md §8.1).

The pre-gate runs *before* retrieval and the LLM: a question that must be refused
never reaches Chroma, never loads the model, and never gets persisted. PII is
detected by pattern only, and the match is never echoed into an artifact.

Output validators (V1-V7) belong to §8 and are intentionally not implemented
here yet.
"""

from __future__ import annotations

import re

from config.config import AppConfig
from src.lexicon import Lexicon, load_lexicon
from src.models import Decision

#: Checked in this order; the first match wins. PAN/Aadhaar/mobile are
#: deliberately anchored so a scheme name or a 4-digit amount never matches.
PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pan", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.I)),
    ("aadhaar", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("mobile", re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}", re.I)),
    ("otp", re.compile(r"\botp\b|\bone[- ]?time password\b", re.I)),
)

#: Refusal groups in the order §8.1 checks them. The lexicon already owns the
#: patterns; this list only fixes the precedence.
REFUSAL_ORDER: tuple[tuple[Decision, tuple[str, ...]], ...] = (
    ("refuse_advice", ("advice", "portfolio")),
    ("refuse_performance", ("performance",)),
    ("refuse_out_of_scope", ("out_of_scope",)),
)


def detect_pii(text: str) -> str | None:
    """Return the PII kind that matched, or None. Never returns the value."""
    for kind, pattern in PII_PATTERNS:
        if pattern.search(text):
            return kind
    return None


def _matches(patterns: dict[str, list[re.Pattern]], groups: tuple[str, ...],
             text: str) -> bool:
    return any(pattern.search(text)
               for group in groups
               for pattern in patterns.get(group, []))


def pre_gate(query: str, cfg: AppConfig, lexicon: Lexicon | None = None) -> Decision:
    """Classify a question before spending a retrieval or an LLM call."""
    if detect_pii(query):
        return "refuse_pii"

    lex = lexicon or load_lexicon(cfg.resolve(cfg.corpus.lexicon))
    for decision, groups in REFUSAL_ORDER:
        if _matches(lex.refusal, groups, query):
            return decision
    return "answer"


def pick_education_link(cfg: AppConfig) -> str | None:
    """One education URL from the manifest, if the corpus has one.

    This corpus intentionally has none (the user supplied only 5 scheme/AMC
    pages). Returning None is honest: a refusal must not invent a link.
    """
    from src.loader import read_manifest                 # lazy: avoids a heavy import

    rows = read_manifest(cfg)
    return next((row["url"] for row in rows
                 if row.get("education", "").lower() == "true"), None)

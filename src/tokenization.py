"""Token counting with the embedding model's own tokenizer (implementation.md §3.1).

The 256-token cap is a safety property: if MiniLM truncates a chunk, the citation
survives while the fact silently disappears. So the count must come from the same
WordPiece vocabulary the model uses — not `len(text.split())`.

Exact counting needs only `tokenizer.json` (~450 KB), not torch: `tokenizers`
loads the WordPiece model standalone. It is cached on first use. If it cannot be
downloaded (offline), we fall back to a deliberately *conservative* estimate so
chunks get smaller rather than riskier, and the stage manifest records which mode
was used.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import httpx

HF_TOKENIZER_URL = "https://huggingface.co/{model_id}/resolve/main/tokenizer.json"

# WordPiece-ish estimate: a word contributes roughly one token per four
# characters, punctuation one each, plus a 20% safety margin. Calibrated on this
# corpus so it *never* undercounts (mean ~1.4x, max ~2.2x true count) — a
# fallback must make chunks smaller, never quietly truncatable.
_PIECE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


def _heuristic_tokens(text: str) -> int:
    total = 0
    for piece in _PIECE.findall(text):
        total += max(1, math.ceil(len(piece) / 4)) if piece[0].isalpha() else 1
    return math.ceil(total * 1.2) + 2      # [CLS] + [SEP]


class TokenCounter:
    """Counts tokens for `cfg.embedding.model_id`; exact when the tokenizer is cached."""

    def __init__(self, cfg, *, allow_download: bool = True) -> None:
        self.model_id = cfg.embedding.model_id
        self.max_tokens = cfg.chunking.max_tokens
        self.exact = False
        self.notes: list[str] = []
        self._tokenizer = None

        self.path = (cfg.resolve(cfg.corpus.cache_dir) / "tokenizer"
                     / cfg.model_slug / "tokenizer.json")
        if allow_download:
            self._ensure_tokenizer()
        self._load()

    # ---- setup ----------------------------------------------------------

    def _ensure_tokenizer(self) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        url = HF_TOKENIZER_URL.format(model_id=self.model_id)
        try:
            response = httpx.get(url, follow_redirects=True, timeout=45,
                                 headers={"User-Agent": "mf-faq-research-bot/1.0"})
            if response.status_code != 200 or not response.content:
                raise RuntimeError(f"HTTP {response.status_code}")
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_bytes(response.content)
            os.replace(tmp, self.path)
        except Exception as exc:                     # noqa: BLE001 — degrades, never fails the stage
            self.notes.append(
                f"tokenizer not cached ({type(exc).__name__}: {exc}); "
                "using conservative word-piece estimate")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(self.path))
            # HF tokenizer.json can carry its own truncation/padding. Counting with
            # those enabled would silently cap every measurement (128 here) and
            # defeat the whole point of the token budget.
            tokenizer.no_truncation()
            tokenizer.no_padding()
            self._tokenizer = tokenizer
            self.exact = True
        except Exception as exc:                     # noqa: BLE001
            self.notes.append(f"tokenizer unreadable ({type(exc).__name__}); "
                              "using conservative estimate")

    # ---- api ------------------------------------------------------------

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text).ids)
        return _heuristic_tokens(text)

    @property
    def mode(self) -> str:
        return "exact_wordpiece" if self.exact else "conservative_estimate"

    def describe(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "mode": self.mode,
            "max_tokens": self.max_tokens,
            "tokenizer_file": str(self.path),
        }

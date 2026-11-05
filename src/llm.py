"""Optional Groq rephrasing layer over an already-verified extractive claim.

Design rule: the extractive template is the source of truth for numbers, names
and qualifiers. The LLM may only rewrite that sentence. Its output is treated as
untrusted: it must pass the same citation/sentence/advice/PII/value validators as
a template answer, plus a fact-preservation check, or the caller keeps the
extractive text and records the fallback.

No key, no model id, an SDK error, malformed JSON, or a timeout all degrade to
the extractive path. The UI never needs to crash because the optional layer is
unavailable.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from config.config import AppConfig, ConfigError

PROVIDER = "groq"
API_KEY_ENV = "GROQ_API_KEY"
MAX_SENTENCES = 3

SYSTEM_PROMPT = (
    "You rewrite ONE verified mutual-fund fact in plain English. "
    "Return JSON only, with exactly one key: {\"answer\": \"<rewrite>\"}. "
    "Rules: "
    "(1) Preserve every number, unit, percentage, currency amount, fund name, "
    "plan name and qualifier exactly as written. "
    "(2) Do not add, remove or infer any fact. "
    "(3) Use at most 3 sentences; one sentence is preferred. "
    "(4) Do not give advice, opinions, comparisons, predictions or performance "
    "claims. "
    "(5) Do not include URLs, links, citations or markdown. "
    "(6) If the fact cannot be rewritten safely, repeat it verbatim."
)

USER_PROMPT = (
    "Verified fact:\n{claim}\n\n"
    "Return only the JSON object described in the system instructions. "
    "Keep the answer grounded in the verified fact above."
)

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


@dataclass
class LLMResult:
    """One rephrasing attempt, including all data the trace panel needs."""

    answer: str = ""
    provider: str = PROVIDER
    model: str = ""
    prompt: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    parsed: dict[str, Any] | None = None
    latency_ms: int = 0
    usage: dict[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.answer and not self.error)


class GroqPhraser:
    """Thin Groq chat-completions client behind a one-method interface."""

    provider = PROVIDER

    def __init__(self, cfg: AppConfig, *, client: Any | None = None):
        self.cfg = cfg
        self.model = cfg.llm.model
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from groq import Groq                 # lazy: only needed when enabled

            self._client = Groq(
                api_key=os.getenv(API_KEY_ENV, "").strip(),
                timeout=self.cfg.llm.timeout_s,
                max_retries=self.cfg.llm.max_retries,
            )
        return self._client

    def rephrase(self, claim: str, *, expected_value: str | None = None,
                 scheme_name: str | None = None,
                 chunk_id: str | None = None) -> LLMResult:
        """Rephrase ``claim``. Never raises: failures come back as ``error``."""
        prompt = {
            "system": SYSTEM_PROMPT,
            "user": USER_PROMPT.format(claim=claim),
            "context_chunk_ids": [chunk_id] if chunk_id else [],
        }
        started = time.monotonic()
        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt["user"]},
                ],
                "temperature": self.cfg.llm.temperature,
                "top_p": self.cfg.llm.top_p,
                "max_tokens": self.cfg.llm.max_tokens,
                "timeout": self.cfg.llm.timeout_s,
            }
            if self.cfg.llm.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:                  # network, auth, JSON-mode 400
            return LLMResult(
                model=self.model, prompt=prompt,
                latency_ms=int((time.monotonic() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        try:
            choice = response.choices[0]
            text = choice.message.content or ""
        except Exception as exc:
            return LLMResult(model=self.model, prompt=prompt,
                             latency_ms=latency_ms,
                             error=f"unreadable_response: {exc}")

        usage = None
        try:
            usage = (response.usage.model_dump()
                     if response.usage is not None else None)
        except Exception:
            usage = None

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("top-level JSON is not an object")
            answer = str(parsed.get("answer", "")).strip()
            if not answer:
                raise ValueError("missing non-empty 'answer'")
        except (json.JSONDecodeError, ValueError) as exc:
            return LLMResult(
                model=self.model, prompt=prompt, text=text,
                latency_ms=latency_ms, usage=usage,
                error=f"invalid_json: {exc}",
            )

        return LLMResult(
            answer=answer, model=self.model, prompt=prompt, text=text,
            parsed=parsed, latency_ms=latency_ms, usage=usage,
        )


def build_phraser(cfg: AppConfig) -> GroqPhraser | None:
    """Return a phraser only when the provider, model id and key are present."""
    if cfg.llm.provider != PROVIDER:
        return None
    try:
        model = cfg.llm.model
    except ConfigError:
        return None
    if not model or not os.getenv(API_KEY_ENV, "").strip():
        return None
    return GroqPhraser(cfg)


def numbers_in(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text or ""))


def fact_preserved(claim: str, answer: str, *,
                   expected_value: str | None = None,
                   scheme_name: str | None = None) -> bool:
    """Conservative check that a rewrite kept the facts it was given."""
    if not answer.strip():
        return False
    if expected_value and expected_value not in answer:
        return False
    if numbers_in(claim) - numbers_in(answer):
        return False
    if scheme_name and scheme_name != "the scheme":
        if scheme_name.casefold() not in answer.casefold():
            return False
    if URL_RE.search(answer):
        return False
    return True

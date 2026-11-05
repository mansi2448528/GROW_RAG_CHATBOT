"""Lexicon loading + matching (implementation.md §3.1, §6).

Patterns live in `config/lexicon.yaml` because they change often; Python only
implements the matching. Fact keys are a closed vocabulary: retrieval filters and
the guardrail depend on exact key names, so free-text keys are impossible here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from config.config import ConfigError


@dataclass(frozen=True)
class Lexicon:
    fact_patterns: dict[str, list[re.Pattern]]
    fact_queries: dict[str, str]
    scheme_patterns: dict[str, list[re.Pattern]]
    scheme_names: dict[str, str]
    refusal: dict[str, list[re.Pattern]]

    @property
    def fact_vocabulary(self) -> tuple[str, ...]:
        return tuple(self.fact_patterns)

    def fact_keys(self, text: str) -> list[str]:
        """Canonical keys mentioned in `text`, in vocabulary order."""
        return [key for key, patterns in self.fact_patterns.items()
                if any(p.search(text) for p in patterns)]

    def schemes(self, text: str) -> list[str]:
        return [key for key, patterns in self.scheme_patterns.items()
                if any(p.search(text) for p in patterns)]


def _compiled(section: dict, name: str, path: Path) -> dict[str, list[re.Pattern]]:
    if not isinstance(section, dict) or not section:
        raise ConfigError(f"{path} is missing '{name}'")
    compiled: dict[str, list[re.Pattern]] = {}
    for key, patterns in section.items():
        if isinstance(patterns, str):
            patterns = [patterns]
        try:
            compiled[key] = [re.compile(p, re.I) for p in patterns]
        except re.error as exc:
            raise ConfigError(f"{path}: bad regex for {name}.{key}: {exc}") from exc
    return compiled


def load_lexicon(path: Path) -> Lexicon:
    if not path.exists():
        raise ConfigError(f"lexicon not found: {path}")
    raw = yaml.safe_load(path.read_text("utf-8")) or {}
    fact_patterns = _compiled(raw.get("fact_keys"), "fact_keys", path)
    fact_queries = {str(k): str(v) for k, v in (raw.get("fact_queries") or {}).items()}
    unknown = sorted(set(fact_queries) - set(fact_patterns))
    if unknown:
        raise ConfigError(f"{path}: fact_queries has unknown key(s): {', '.join(unknown)}")
    return Lexicon(
        fact_patterns=fact_patterns,
        fact_queries=fact_queries,
        scheme_patterns=_compiled(raw.get("schemes"), "schemes", path),
        scheme_names=dict(raw.get("scheme_names") or {}),
        refusal=_compiled(raw.get("refusal"), "refusal", path),
    )

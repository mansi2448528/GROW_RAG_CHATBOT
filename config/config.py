"""Typed configuration loader.

implementation.md §1: all model ids, sizes, thresholds and paths live in
config.yaml. Nothing downstream may hardcode them.

Two validations are deliberately fatal at startup:
  1. chunking.max_tokens > embedding.max_seq_length  -> silent truncation would
     drop facts while keeping the citation.
  2. embedding.dim inconsistent with a known model id -> vector space mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_VERSION = "1.0.0"

# Models whose output dimension we can verify up-front. An unknown model id is
# allowed (configurable stack) but dim must then be supplied explicitly.
KNOWN_EMBEDDING_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
}


class ConfigError(RuntimeError):
    """Raised for a config that is structurally valid YAML but unusable."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppSection(_Section):
    name: str = "MF FAQ Assistant"
    artifacts_dir: Path = Path("data/artifacts")
    log_level: str = "INFO"


class CorpusSection(_Section):
    manifest: Path = Path("data/corpus/urls.csv")
    lexicon: Path = Path("config/lexicon.yaml")
    cache_dir: Path = Path("data/cache")
    min_chars_per_page: int = Field(1500, gt=0)
    # Structured extraction (scheme pages) is deliberately terse: a compact fact
    # sheet is the goal, not the whole page. Missing facts are caught by the
    # parser's "facts not found" warning, not by a character count.
    min_chars_by_page_type: dict[str, int] = Field(default_factory=dict)
    request_timeout_s: float = Field(20, gt=0)
    retries: int = Field(1, ge=0, le=5)
    rate_limit_rps: float = Field(0.5, gt=0)
    user_agent: str = "mf-faq-research-bot/1.0 (+public pages only)"

    def min_chars(self, page_type: str) -> int:
        return self.min_chars_by_page_type.get(page_type, self.min_chars_per_page)


class EmbeddingSection(_Section):
    model_id: str
    dim: int = Field(gt=0)
    normalize: bool = True
    max_seq_length: int = Field(256, gt=0)
    batch_size: int = Field(32, gt=0)
    device: str = "cpu"
    preprocessing: str = "page_title+section_path"

    @model_validator(mode="after")
    def _dim_matches_known_model(self) -> "EmbeddingSection":
        known = KNOWN_EMBEDDING_DIMS.get(self.model_id)
        if known is not None and known != self.dim:
            raise ValueError(
                f"embedding.dim={self.dim} but {self.model_id} produces {known} dims"
            )
        return self


class VectorStoreSection(_Section):
    provider: Literal["chroma"] = "chroma"
    persist_dir: Path = Path("data/chroma")
    collection_prefix: str = "mf_faq"
    distance: Literal["cosine", "l2", "ip"] = "cosine"


class ChunkingSection(_Section):
    max_tokens: int = Field(256, gt=0)
    merge_target_tokens: tuple[int, int] = (200, 400)
    overlap_ratio: float = Field(0.12, ge=0, lt=0.5)
    tokenizer_model: str | None = None
    quarantine_patterns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _merge_range(self) -> "ChunkingSection":
        lo, hi = self.merge_target_tokens
        if lo <= 0 or hi < lo:
            raise ValueError("chunking.merge_target_tokens must be [low, high] with 0 < low <= high")
        return self


class RetrievalSection(_Section):
    top_k: int = Field(5, ge=1, le=50)
    keep: int = Field(3, ge=1)
    score_threshold: float = Field(0.35, ge=0, le=1)
    lexical_boost: float = Field(0.15, ge=0, le=1)
    filter_by_scheme: bool = True
    exclude_quarantined: bool = True
    rewrite: Literal["template", "llm"] = "template"

    @model_validator(mode="after")
    def _keep_le_top_k(self) -> "RetrievalSection":
        if self.keep > self.top_k:
            raise ValueError("retrieval.keep must be <= retrieval.top_k")
        return self


class LLMSection(_Section):
    provider: Literal["groq", "openai", "gemini", "anthropic", "mistral"] = "groq"
    model_env: str = "LLM_MODEL"
    temperature: float = Field(0.1, ge=0, le=1)
    top_p: float = Field(0.9, ge=0, le=1)
    max_tokens: int = Field(220, gt=0)
    timeout_s: float = Field(20, gt=0)
    max_retries: int = Field(1, ge=0, le=2)
    json_mode: bool = True

    @cached_property
    def model(self) -> str:
        value = os.getenv(self.model_env, "").strip()
        if not value:
            raise ConfigError(
                f"{self.model_env} is not set. Copy .env.example to .env and set a "
                f"model id your {self.provider} account can serve."
            )
        return value


class GuardrailsSection(_Section):
    use_llm_judge: bool = False
    max_sentences: int = Field(3, ge=1, le=5)
    advice_terms: list[str] = Field(default_factory=list)


class ArtifactsSection(_Section):
    write_per_query: bool = True
    qa_log: Path = Path("data/artifacts/qa_log.csv")


class AppConfig(_Section):
    app: AppSection
    corpus: CorpusSection
    embedding: EmbeddingSection
    vectorstore: VectorStoreSection
    chunking: ChunkingSection
    retrieval: RetrievalSection
    llm: LLMSection
    guardrails: GuardrailsSection
    artifacts: ArtifactsSection

    @model_validator(mode="after")
    def _chunk_fits_model(self) -> "AppConfig":
        if self.chunking.max_tokens > self.embedding.max_seq_length:
            raise ValueError(
                f"chunking.max_tokens={self.chunking.max_tokens} exceeds "
                f"embedding.max_seq_length={self.embedding.max_seq_length}; "
                "MiniLM would truncate the chunk and drop facts silently"
            )
        return self

    # ---- derived values -------------------------------------------------

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    @property
    def model_slug(self) -> str:
        tail = self.embedding.model_id.split("/")[-1].lower()
        return "".join(c if c.isalnum() else "_" for c in tail).strip("_")

    @cached_property
    def fingerprint(self) -> str:
        """Binds a Chroma collection to the model that produced its vectors."""
        raw = "|".join([
            self.embedding.model_id,
            str(self.embedding.dim),
            str(self.embedding.normalize).lower(),
            self.embedding.preprocessing,
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    @cached_property
    def collection_name(self) -> str:
        return f"{self.vectorstore.collection_prefix}_{self.model_slug}_{self.fingerprint}"

    @cached_property
    def config_hash(self) -> str:
        payload = self.model_dump(mode="json")
        raw = json.dumps(payload, sort_keys=True) + PIPELINE_VERSION
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def stage_dir(self, stage: str) -> Path:
        return self.resolve(self.app.artifacts_dir) / stage


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML + .env into a validated AppConfig."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        raise ConfigError(f"config not found: {cfg_path}")

    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text("utf-8")) or {}
    try:
        cfg = AppConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config at {cfg_path}:\n{exc}") from exc

    # Hosted platforms (Vercel container Functions, Render, etc.) mount the
    # source tree read-only and provide a writable /tmp. These overrides let
    # the launcher redirect state without changing the checked-in config.
    path_overrides = (
        ("MF_FAQ_ARTIFACTS_DIR", cfg.app, "artifacts_dir"),
        ("MF_FAQ_CHROMA_DIR", cfg.vectorstore, "persist_dir"),
        ("MF_FAQ_CACHE_DIR", cfg.corpus, "cache_dir"),
    )
    for env_name, section, attr in path_overrides:
        override = os.getenv(env_name, "").strip()
        if override:
            setattr(section, attr, Path(override))

    for rel in (cfg.app.artifacts_dir, cfg.vectorstore.persist_dir,
                cfg.corpus.cache_dir, cfg.corpus.manifest.parent):
        cfg.resolve(rel).mkdir(parents=True, exist_ok=True)

    return cfg


def summary(cfg: AppConfig) -> dict[str, Any]:
    """Flat view for logs, --check-config and artifact manifests."""
    return {
        "pipeline_version": PIPELINE_VERSION,
        "config_hash": cfg.config_hash,
        "embedding": f"{cfg.embedding.model_id} ({cfg.embedding.dim}-d, max {cfg.embedding.max_seq_length} tokens)",
        "fingerprint": cfg.fingerprint,
        "collection": cfg.collection_name,
        "chunking": f"max {cfg.chunking.max_tokens} tokens, merge {cfg.chunking.merge_target_tokens}",
        "retrieval": f"top_k={cfg.retrieval.top_k} keep={cfg.retrieval.keep} thr={cfg.retrieval.score_threshold}",
        "llm": f"{cfg.llm.provider} (model from ${cfg.llm.model_env})",
        "manifest": str(cfg.resolve(cfg.corpus.manifest)),
    }


if __name__ == "__main__":
    for key, value in summary(load_config()).items():
        print(f"{key:18} {value}")

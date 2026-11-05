"""Manifest -> Document -> disk, and the read-back used by later stages.

implementation.md §2.4 / §2.5. Loading is a stage with a written artifact, not a
side effect of scraping: once 01_loaded/ exists, chunking never touches the
network.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from datetime import date
from pathlib import Path

from config.config import AppConfig, ConfigError
from src.artifacts import StageRecorder
from src.fetcher import HttpFetcher
from src.models import Document, DocMeta
from src.parser import parse

MANIFEST_COLUMNS = ("id", "url", "page_type", "scheme", "amc", "plan", "intent_tags", "education")
PAGE_TYPES = {"groww_scheme", "groww_amc", "amc_page", "sebi_page", "amfi_page"}
MIN_URLS, MAX_URLS = 15, 25


class PipelineError(RuntimeError):
    """A stage's input is missing or unusable."""


def read_manifest(cfg: AppConfig) -> list[dict[str, str]]:
    path = cfg.resolve(cfg.corpus.manifest)
    if not path.exists():
        raise ConfigError(f"corpus manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ConfigError(f"{path.name} is empty")
    missing = [c for c in MANIFEST_COLUMNS if c not in rows[0]]
    if missing:
        raise ConfigError(f"{path.name} is missing columns: {', '.join(missing)}")
    return [{k: (v or "").strip() for k, v in row.items()} for row in rows]


def validate_manifest(rows: list[dict[str, str]], lexicon_schemes: set[str] | None = None) -> list[str]:
    """Hard errors raise; soft issues are returned as TODO notes."""
    problems: list[str] = []
    notes: list[str] = []

    urls = [row["url"] for row in rows]
    problems += [f"duplicate url: {url}" for url in sorted({u for u in urls if urls.count(u) > 1})]
    for i, row in enumerate(rows, start=2):
        if not row["url"].startswith("https://"):
            problems.append(f"row {i}: url must be https ({row['url'][:40]})")
        if row["page_type"] not in PAGE_TYPES:
            problems.append(f"row {i}: unknown page_type '{row['page_type']}'")
        if not row["id"]:
            problems.append(f"row {i}: empty id")
        if lexicon_schemes and row["scheme"] and row["scheme"] not in lexicon_schemes:
            problems.append(f"row {i}: scheme '{row['scheme']}' not in lexicon.yaml")

    ids = [row["id"] for row in rows]
    problems += [f"duplicate id: {dup}" for dup in sorted({d for d in ids if ids.count(d) > 1})]

    if len(rows) < MIN_URLS:
        notes.append(f"corpus incomplete: {len(rows)}/{MIN_URLS} minimum URLs (§2.1)")
    if len(rows) > MAX_URLS:
        problems.append(f"manifest has {len(rows)} rows; the brief allows at most {MAX_URLS}")
    if sum(1 for row in rows if row["education"].lower() == "true") < 2:
        notes.append("need >= 2 rows with education=true for refusal links")

    if problems:
        raise ConfigError("invalid manifest:\n  - " + "\n  - ".join(problems))
    return notes


def sha256_blocks(blocks) -> str:
    payload = json.dumps([b.text for b in blocks], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_all(cfg: AppConfig, rec: StageRecorder, only: str | None = None,
             use_cache: bool = True) -> list[Document]:
    rows = read_manifest(cfg)
    for note in validate_manifest(rows):
        rec.warnings.append(note)

    # A full load owns the folder: stale documents from a previous manifest must
    # not survive into chunking (they would be cited even though the URL is gone).
    if not only:
        for stale in cfg.stage_dir("01_loaded").glob("*.jsonl"):
            stale.unlink()

    fetcher = HttpFetcher(cfg)
    docs: list[Document] = []
    report: list[dict] = []

    try:
        for row in rows:
            if only and row["id"] != only:
                continue

            meta = DocMeta(
                doc_id=row["id"], source_url=row["url"], page_title="",
                page_type=row["page_type"], scheme=row["scheme"] or None,
                amc=row["amc"], plan=row["plan"] or "na",
                education=row["education"].lower() == "true",
            )

            fetched = fetcher.get(row["url"], use_cache=use_cache)
            meta.fetched_at = fetched.fetched_at

            blocks, warnings = parse(fetched.html, meta)
            meta.page_title = _page_title(fetched.html) or row["id"]

            chars = sum(len(b.text) for b in blocks)
            minimum = cfg.corpus.min_chars(meta.page_type)
            status = "ERROR" if not blocks else ("SUSPECT" if chars < minimum else "OK")

            doc = Document(meta=meta, blocks=blocks, content_sha256=sha256_blocks(blocks),
                           http_status=fetched.status, parser=f"{row['page_type']}@v1",
                           warnings=warnings + ([fetched.error] if fetched.error else []))
            docs.append(doc)
            rec.jsonl("01_loaded", f"{row['id']}.jsonl", [doc.to_dict()])

            fact_keys = sorted({b.fact_key for b in blocks if b.fact_key})
            report.append({
                "doc_id": row["id"], "page_type": row["page_type"], "url": row["url"],
                "status": status, "http": fetched.status, "cached": fetched.from_cache,
                "chars": chars, "blocks": len(blocks), "facts": "|".join(fact_keys),
                "warnings": "; ".join(warnings), "error": fetched.error or "",
            })
            print(f"  {status:8} {row['id']:22} {len(blocks):3} blocks {chars:6} chars "
                  f"{'[cache]' if fetched.from_cache else ''}")
    finally:
        fetcher.close()

    rec.csv("01_loaded", "load_report.csv", report, columns=[
        "doc_id", "page_type", "url", "status", "http", "cached",
        "chars", "blocks", "facts", "warnings", "error"])
    rec.markdown("01_loaded", "load_summary.md", _summary_md(report))
    rec.manifest("01_loaded", {
        "documents": len(docs),
        "ok": sum(1 for r in report if r["status"] == "OK"),
        "suspect": sum(1 for r in report if r["status"] == "SUSPECT"),
        "error": sum(1 for r in report if r["status"] == "ERROR"),
        "blocks": sum(r["blocks"] for r in report),
        "chars": sum(r["chars"] for r in report),
    })
    return docs


def load_documents(cfg: AppConfig) -> list[Document]:
    """Read 01_loaded/ back into Documents — the only input to chunking."""
    folder = cfg.stage_dir("01_loaded")
    files = sorted(folder.glob("*.jsonl"))
    if not files:
        raise PipelineError("01_loaded is empty — run: python -m app.pipeline --stage load")

    docs: list[Document] = []
    for path in files:
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                docs.append(Document.from_jsonl(line))
    if not docs:
        raise PipelineError("01_loaded contains no documents — run: python -m app.pipeline --stage load")
    return docs


def _page_title(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    return (soup.title.string or "").strip() if soup.title and soup.title.string else ""


def _summary_md(report: list[dict]) -> str:
    lines = ["# Load report", "",
             "| doc_id | page_type | status | blocks | chars | facts |",
             "|---|---|---|---|---|---|"]
    for row in report:
        lines.append(f"| {row['doc_id']} | {row['page_type']} | {row['status']} | "
                     f"{row['blocks']} | {row['chars']} | {row['facts']} |")
    warnings = [f"- **{row['doc_id']}**: {row['warnings']}" for row in report if row["warnings"]]
    if warnings:
        lines += ["", "## Warnings", *warnings]
    return "\n".join(lines)

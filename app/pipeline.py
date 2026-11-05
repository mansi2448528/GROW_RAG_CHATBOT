"""Pipeline CLI — implemented stage by stage per implementation.md.

§1  --check-config                 validate config.yaml + corpus manifest
§2  --stage load [--only ID]       scrape + parse -> data/artifacts/01_loaded/
§3  --stage chunk                  atomic-fact chunks -> 02_chunks/
§4  --stage embed                  MiniLM vectors  -> 03_embeddings/
§5  --stage store [--rebuild]      Chroma collection -> 04_vectordb/
§6  --query "..."                  retrieval trace     -> 05_retrieval/
§6  --eval-retrieval               eval/probes.csv, hit@3 >= 8/10
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

from config.config import AppConfig, ConfigError, load_config, summary
from src.artifacts import StageRecorder
from src.chunker import chunk_all, lockin_chunk
from src.embedder import EmbeddingError, embed_all
from src.loader import PipelineError, load_all, read_manifest, validate_manifest
from src.retriever import answer_query, evaluate_retrieval, make_query_id
from src.store import StoreError, store_all

STAGES = ("load", "chunk", "embed", "store")
IMPLEMENTED = ("load", "chunk", "embed", "store")
RETRIEVAL_ERRORS = (ConfigError, PipelineError, StoreError, EmbeddingError)


def check_config(cfg: AppConfig) -> int:
    print("CONFIG")
    for key, value in summary(cfg).items():
        print(f"  {key:16} {value}")

    rows = read_manifest(cfg)
    counts = Counter(row["page_type"] for row in rows)
    schemes = sorted({row["scheme"] for row in rows if row["scheme"]})
    education = sum(1 for row in rows if row["education"].lower() == "true")

    print("\nMANIFEST")
    print(f"  rows             {len(rows)}")
    for page_type, n in sorted(counts.items()):
        print(f"  {page_type:16} {n}")
    print(f"  schemes          {', '.join(schemes) or '—'}")
    print(f"  education links  {education}")

    notes = validate_manifest(rows)
    if notes:
        print("\nTODO")
        for note in notes:
            print(f"  - {note}")

    print("\nOK — config valid, manifest structurally sound"
          + (" (corpus still incomplete)" if notes else ""))
    return 0


def run_query(cfg: AppConfig, query: str) -> int:
    """§6: one question through the gate + retrieval, with a written trace."""
    recorder = StageRecorder(cfg)
    started = time.monotonic()
    try:
        plan = answer_query(query, cfg, recorder=recorder)
    except RETRIEVAL_ERRORS as exc:
        recorder.log_run("query", "FAILED", {}, time.monotonic() - started, str(exc))
        print(f"\nQUERY FAILED: {exc}", file=sys.stderr)
        return 2

    _print_plan(query, plan, cfg)
    recorder.log_run("query", "OK", {
        "decision": plan.decision,
        "outcome": plan.outcome,
        "scheme": plan.scheme or "",
        "fact_keys": ",".join(plan.fact_keys),
        "hits": len(plan.hits),
        "best_score": round(plan.best_score, 4),
    }, time.monotonic() - started)
    return 0


def _print_plan(query: str, plan, cfg: AppConfig) -> None:
    shown = "[redacted: PII]" if plan.decision == "refuse_pii" else query
    print(f"\nQUERY  {shown}")
    print(f"GATE   decision={plan.decision} · outcome={plan.outcome}")

    trace = cfg.stage_dir("05_retrieval") / f"{make_query_id(query)}.json"
    if plan.decision != "answer":
        print(f"  education {plan.education_url or '—'}")
        print("  trace     not written (PII is never persisted)"
              if plan.decision == "refuse_pii" else f"  trace     {trace}")
        return

    print("ANALYSE")
    print(f"  scheme    {plan.scheme or '—'}")
    print(f"  fact_keys {', '.join(plan.fact_keys) or '—'}")
    print(f"  rewrite   {plan.rewrite!r}")
    print(f"  filters   {plan.filters}")
    if plan.filters.get("ambiguous_schemes"):
        print("  note      multiple schemes named — clarify before answering")

    if plan.hits:
        print("\nHITS  (cosine + lexical boost, one per source URL)")
        print("  #  chunk_id                 score  rank  doc_id")
        for rank, hit in enumerate(plan.hits, start=1):
            print(f"  {rank}  {hit.chunk_id:23} {hit.score:.3f}  "
                  f"{hit.rank_score:.3f}  {hit.doc_id}")
            print(f"     {hit.section or '—'} · {hit.source_url}")
    else:
        print("\nHITS  none above threshold")
    print(f"\nTRACE  {trace}")


def run_eval_retrieval(cfg: AppConfig) -> int:
    """§6: hit@3 on eval/probes.csv; nonzero when below the 8/10 target."""
    recorder = StageRecorder(cfg)
    started = time.monotonic()
    try:
        result = evaluate_retrieval(cfg, recorder=recorder)
    except RETRIEVAL_ERRORS as exc:
        recorder.log_run("retrieval", "FAILED", {}, time.monotonic() - started,
                         str(exc))
        print(f"\nRETRIEVAL EVAL FAILED: {exc}", file=sys.stderr)
        return 2

    print("\nRETRIEVAL EVAL")
    print(f"  probes   {result.total}")
    print(f"  hit@3    {result.passed}/{result.total} "
          f"({result.hit_at_3:.0%}) · target ≥ {result.target}")
    print("\n  #  query_id              expected                       "
          "predicted                      rank  score  outcome")
    for index, row in enumerate(result.rows, start=1):
        mark = "✓" if row["ok"] else "✗"
        print(f"  {index:2} {row['query_id']:21} {row['expected']:30} "
              f"{row['predicted']:30} {str(row['rank'] or '—'):4}  "
              f"{row['score']:.3f}  {row['outcome']} {mark}")

    print(f"\n  report   {result.report_path}")
    print(f"  traces   {cfg.stage_dir('05_retrieval')}")

    recorder.log_run("retrieval", "OK" if result.ok else "BELOW_TARGET", {
        "probes": result.total,
        "passed": result.passed,
        "hit_at_3": round(result.hit_at_3, 4),
        "target": result.target,
    }, time.monotonic() - started)

    if not result.ok:
        print(f"\nbelow target: {result.passed}/{result.total} "
              f"(need {result.target})", file=sys.stderr)
        return 1
    return 0


def run_stage(cfg: AppConfig, stage: str, args) -> int:
    recorder = StageRecorder(cfg)
    started = time.monotonic()
    counts: dict[str, int] = {}

    try:
        if stage in ("load", "chunk", "embed", "store", "all"):
            chunks = None
            if stage in ("load", "all"):
                print(f"LOAD  run {recorder.run_id}")
                docs = load_all(cfg, recorder, only=args.only, use_cache=not args.no_cache)
                counts = {"documents": len(docs),
                          "ok": sum(1 for d in docs if not d.warnings),
                          "blocks": sum(len(d.blocks) for d in docs)}
                _print_fact_coverage(docs)

            if stage in ("chunk", "all"):
                print(f"\nCHUNK run {recorder.run_id}")
                chunks = chunk_all(cfg, recorder)
                counts.update({
                    "chunks": len(chunks),
                    "atomic": sum(1 for c in chunks if c.is_atomic),
                    "quarantined": sum(1 for c in chunks if c.is_quarantined),
                    "max_chunk_tokens": max((c.token_count for c in chunks), default=0),
                })
                _print_chunk_checks(cfg, chunks)

            if stage in ("embed", "all"):
                print(f"\nEMBED run {recorder.run_id}")
                result = embed_all(cfg, recorder, chunks=chunks)
                counts.update({
                    "embedded": len(result.chunks),
                    "dim": int(result.vectors.shape[1]),
                    "fingerprint": cfg.fingerprint,
                    "probes_passed": result.manifest["probes_passed"],
                })
                _print_embed_checks(cfg, result)

            if stage in ("store", "all"):
                print(f"\nSTORE run {recorder.run_id}")
                stored = store_all(cfg, recorder, rebuild=args.rebuild)
                counts.update({
                    "collection": stored.status.name,
                    "records": stored.status.count,
                    "stale_deleted": stored.status.stale_deleted,
                    "rebuilt": stored.status.rebuilt,
                    "min_cosine": round(stored.status.min_cosine, 8),
                })
                _print_store_checks(cfg, stored)

            _print_notes(recorder)

            if stage == "all":
                print("\nnext stages (retrieval eval/UI/guardrails) are not part "
                      "of --stage all — see implementation.md §6–§8")
        else:
            raise PipelineError(f"stage '{stage}' is not implemented yet — "
                                f"done so far: {', '.join(IMPLEMENTED)}")
    except (ConfigError, PipelineError, StoreError) as exc:
        recorder.log_run(stage, "FAILED", counts, time.monotonic() - started, str(exc))
        print(f"\n{stage.upper()} FAILED: {exc}", file=sys.stderr)
        return 2

    duration = time.monotonic() - started
    recorder.log_run(stage, "OK", counts, duration)
    print(f"\n{stage.upper()} done in {duration:.1f}s → "
          f"{cfg.resolve(cfg.app.artifacts_dir)}")
    return 0


def _print_fact_coverage(docs) -> None:
    keys = ("expense_ratio", "exit_load", "min_sip", "lock_in", "riskometer", "benchmark")
    rows = [(d.meta.doc_id, d) for d in docs if d.meta.page_type == "groww_scheme"]
    if not rows:
        return
    print("\nFACT COVERAGE (scheme pages)")
    print("  " + "doc_id".ljust(22) + "".join(k[:7].rjust(9) for k in keys))
    for doc_id, doc in rows:
        present = {b.fact_key for b in doc.blocks if b.fact_key}
        print("  " + doc_id.ljust(22)
              + "".join(("  ✓" if k in present else "  ·").rjust(9) for k in keys))


def _print_chunk_checks(cfg: AppConfig, chunks) -> None:
    limit = cfg.chunking.max_tokens
    biggest = max((c.token_count for c in chunks), default=0)
    print("\nCHUNK CHECKS")
    print(f"  max chunk tokens  {biggest} / {limit}")
    lock = lockin_chunk(chunks)
    print(f"  ELSS lock-in      {lock.chunk_id}: {lock.text}" if lock
          else "  ELSS lock-in      MISSING — the §3 done-check failed")
    print(f"  preview           {cfg.stage_dir('02_chunks') / 'chunks_preview.md'}")


def _print_embed_checks(cfg: AppConfig, result) -> None:
    manifest = result.manifest
    print("\nEMBED CHECKS")
    print(f"  shape             {result.vectors.shape[0]} × {result.vectors.shape[1]} "
          f"{result.vectors.dtype}")
    print(f"  model             {manifest['model_id']}")
    print(f"  fingerprint       {manifest['fingerprint']}  (collection {cfg.collection_name})")
    print(f"  unit length       max |‖v‖-1| = {result.stats['norm_max_deviation']:.2e}")
    print(f"  semantic probes   {manifest['probes_passed']}/{manifest['probes_total']} passed")
    print(f"  sample            {cfg.stage_dir('03_embeddings') / 'sample_dims.md'}")


def _print_store_checks(cfg: AppConfig, stored) -> None:
    status = stored.status
    print("\nSTORE CHECKS")
    print(f"  collection        {status.name}")
    print(f"  records           {status.count} "
          f"({'matches' if status.count == sum(status.per_doc.values()) else 'MISMATCH'} "
          f"chunks) · {status.dim}-d · {status.distance}")
    print(f"  fingerprint       {status.fingerprint}")
    print(f"  stored vs source  min cosine {status.min_cosine:.8f}")
    print(f"  nearest neighbour {'is itself' if status.nearest_is_self else 'MISMATCH'}"
          f" · stale deleted {status.stale_deleted}"
          + (" · rebuilt" if status.rebuilt else ""))
    print(f"  report            {cfg.stage_dir('04_vectordb') / 'collection_report.md'}")


def _print_notes(recorder: StageRecorder) -> None:
    if recorder.warnings:
        print("\nNOTES")
        for note in recorder.warnings:
            print(f"  - {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.pipeline")
    parser.add_argument("--check-config", action="store_true",
                        help="validate config.yaml and the corpus manifest")
    parser.add_argument("--stage", choices=STAGES + ("all",),
                        help="run an ingestion stage")
    parser.add_argument("--only", help="restrict a stage to one doc_id")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore data/cache and re-fetch every URL")
    parser.add_argument("--incremental", action="store_true",
                        help="re-process only documents whose content hash changed")
    parser.add_argument("--rebuild", action="store_true",
                        help="drop the collection and re-ingest from scratch")
    parser.add_argument("--query",
                        help="answer one factual question; writes 05_retrieval/<id>.json")
    parser.add_argument("--eval-retrieval", action="store_true",
                        help="run eval/probes.csv through the §6 retriever (hit@3 ≥ 8/10)")
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.check_config:
        try:
            return check_config(cfg)
        except ConfigError as exc:
            print(f"manifest error: {exc}", file=sys.stderr)
            return 2

    if args.eval_retrieval or args.query is not None:
        if args.eval_retrieval and args.query is not None:
            print("--eval-retrieval and --query are mutually exclusive",
                  file=sys.stderr)
            return 2
        if args.stage:
            print("--query/--eval-retrieval cannot be combined with --stage",
                  file=sys.stderr)
            return 2
        return run_eval_retrieval(cfg) if args.eval_retrieval else run_query(cfg, args.query)

    if args.stage == "all" and args.only:
        print("--only cannot be combined with --stage all: run --stage load --only ID, "
              "then --stage chunk", file=sys.stderr)
        return 2

    if args.stage in ("chunk", "embed", "store") and args.only:
        # All three write a single corpus-wide artifact: a partial run would
        # silently drop docs or misalign rows against chunks.jsonl.
        print(f"--only is not supported for --stage {args.stage} — it writes one "
              "corpus-wide artifact; restrict --stage load instead", file=sys.stderr)
        return 2

    if args.stage:
        return run_stage(cfg, args.stage, args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

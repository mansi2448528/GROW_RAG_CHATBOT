"""Uniform stage writer (implementation.md §1, §2.4).

Every stage writes its own artifact folder plus a manifest, so a reviewer can
tell which config produced which output. All writes are temp-then-rename: a
crashed stage must never leave a half-file for the next stage to ingest.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class StageRecorder:
    def __init__(self, cfg, run_id: str | None = None):
        self.cfg = cfg
        self.run_id = run_id or new_run_id()
        self.warnings: list[str] = []

    def dir(self, stage: str) -> Path:
        path = self.cfg.stage_dir(stage)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---- primitives -----------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        return path

    def jsonl(self, stage: str, name: str, rows: Iterable[dict[str, Any]]) -> Path:
        body = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows)
        return self._atomic_write(self.dir(stage) / name, body + ("\n" if body else ""))

    def npy(self, stage: str, name: str, array: Any) -> Path:
        """Write a numpy array temp-then-rename (binary files get the same care)."""
        import numpy as np

        path = self.dir(stage) / name
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.save(fh, array, allow_pickle=False)
        os.replace(tmp, path)
        return path

    def json(self, stage: str, name: str, obj: Any) -> Path:
        body = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        return self._atomic_write(self.dir(stage) / name, body + "\n")

    def markdown(self, stage: str, name: str, text: str) -> Path:
        return self._atomic_write(self.dir(stage) / name, text.rstrip() + "\n")

    def csv(self, stage: str, name: str, rows: list[dict[str, Any]],
            columns: list[str] | None = None) -> Path:
        columns = columns or (list(rows[0].keys()) if rows else [])
        path = self.dir(stage) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
        return path

    # ---- stage bookkeeping ----------------------------------------------

    def manifest(self, stage: str, counts: dict[str, Any]) -> Path:
        return self.json(stage, "stage_manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "run_id": self.run_id,
            "config_hash": self.cfg.config_hash,
            "collection": self.cfg.collection_name,
            "embedding_fingerprint": self.cfg.fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counts": counts,
            "warnings": self.warnings,
        })

    def log_run(self, stage: str, status: str, counts: dict[str, Any],
                duration_s: float, error: str | None = None) -> Path:
        path = self.cfg.resolve(self.cfg.app.artifacts_dir) / "runs.csv"
        exists = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "run_id", "stage", "status", "duration_s", "config_hash", "counts", "error"])
            if not exists:
                writer.writeheader()
            writer.writerow({
                "run_id": self.run_id, "stage": stage, "status": status,
                "duration_s": round(duration_s, 2), "config_hash": self.cfg.config_hash,
                "counts": json.dumps(counts, default=str), "error": error or "",
            })
        return path

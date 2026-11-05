"""HTTP fetch with on-disk cache (implementation.md §2.2).

Cache-first is deliberate: it makes the demo offline-safe, keeps re-runs
reproducible when a page changes mid-build, and gives the tests fixtures.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx


@dataclass
class FetchResult:
    url: str
    status: int
    html: str = ""
    fetched_at: date = field(default_factory=date.today)
    from_cache: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.html)


class HttpFetcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self._cache_dir = cfg.resolve(cfg.corpus.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            timeout=cfg.corpus.request_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": cfg.corpus.user_agent,
                     "Accept-Language": "en-IN,en;q=0.9"},
        )
        self._last_request = 0.0

    def _paths(self, url: str) -> tuple[Path, Path]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"{key}.html", self._cache_dir / f"{key}.json"

    def get(self, url: str, use_cache: bool = True) -> FetchResult:
        html_path, meta_path = self._paths(url)

        if use_cache and html_path.exists():
            meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
            return FetchResult(
                url=url, status=int(meta.get("status", 200)),
                html=html_path.read_text("utf-8"), from_cache=True,
                fetched_at=date.fromisoformat(meta["fetched_at"]) if meta.get("fetched_at") else date.today(),
            )

        last_error: Exception | None = None
        attempts = self.cfg.corpus.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                self._throttle()
                response = self._client.get(url)
                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response)

                today = date.today()
                html_path.write_text(response.text, encoding="utf-8")
                meta_path.write_text(json.dumps(
                    {"url": url, "status": response.status_code,
                     "fetched_at": today.isoformat()}, indent=2), encoding="utf-8")
                return FetchResult(url=url, status=response.status_code,
                                   html=response.text, fetched_at=today)
            except Exception as exc:            # noqa: BLE001 — surfaced in load_report
                last_error = exc
                if attempt < attempts:
                    time.sleep(1.5 * attempt)

        return FetchResult(url=url, status=0, error=f"{type(last_error).__name__}: {last_error}")

    def _throttle(self) -> None:
        gap = 1.0 / self.cfg.corpus.rate_limit_rps
        wait = gap - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def close(self) -> None:
        self._client.close()

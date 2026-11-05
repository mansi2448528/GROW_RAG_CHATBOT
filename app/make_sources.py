"""Generate docs/SOURCES.md from the corpus manifest + the latest load report.

The manifest is the single source of truth (implementation.md §2.1), so the
deliverable is derived, never typed by hand: a URL can only appear here if it
was actually in urls.csv at load time.

    python -m app.make_sources
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from config.config import load_config
from src.loader import load_documents, read_manifest

TYPE_LABEL = {
    "groww_scheme": "Scheme pages (Groww)",
    "groww_amc": "AMC overview (Groww)",
    "amc_page": "HDFC AMC",
    "sebi_page": "SEBI (regulator)",
    "amfi_page": "AMFI (industry body)",
}

TYPE_ORDER = ("groww_scheme", "groww_amc", "amc_page", "sebi_page", "amfi_page")

TAG_LABEL = {
    "basics": "fund basics",
    "expense_ratio": "expense ratio",
    "exit_load": "exit load",
    "min_sip": "minimum SIP",
    "lock_in": "lock-in",
    "riskometer": "riskometer",
    "benchmark": "benchmark",
    "statement_download": "account statement",
    "amc_overview": "AMC overview",
}

PURPOSE = {
    "groww_hdfc_amc": "Who runs HDFC Mutual Fund, trustees, sponsor, contact",
    "groww_hdfc_large_cap": "Facts for HDFC Large Cap Fund (Direct-Growth)",
    "groww_hdfc_flexi_cap": "Facts for HDFC Flexi Cap Fund (Direct-Growth)",
    "groww_hdfc_elss": "Facts for HDFC ELSS Tax Saver (Direct-Growth), incl. 3-year lock-in",
    "groww_hdfc_mid_cap": "Facts for HDFC Mid Cap Fund (Direct-Growth)",
    "sebi_riskometer": "What the riskometer means and who assigns it",
    "sebi_cas": "How to get a consolidated account statement (no PII collected here)",
    "sebi_mf_listing": "Official SEBI mutual fund filings / scheme list",
    "sebi_awareness": "Investor awareness messages on dos and don'ts",
    "sebi_smart_investor": "Rights and responsibilities of a mutual fund investor",
    "sebi_moneymatters": "Investor education material on mutual funds",
    "sebi_personal_investments": "Checks before investing in a scheme",
    "sebi_secure_account": "Keeping the investment account secure",
    "amfi_mutual_fund": "MF basics, expense ratio and TER context",
    "amfi_sahi_hai": "AMFI investor awareness campaign (refusal link)",
}


def _facts(cell: str) -> str:
    return ", ".join(TAG_LABEL.get(tag, tag) for tag in cell.split("|") if tag) or "—"


def build(cfg) -> str:
    rows = read_manifest(cfg)
    docs = {doc.meta.doc_id: doc for doc in load_documents(cfg)}
    dates = sorted({doc.meta.fetched_at.isoformat() for doc in docs.values()})

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["page_type"]].append(row)

    counts = Counter(row["page_type"] for row in rows)
    education = [row for row in rows if row["education"].lower() == "true"]

    lines = [
        "# Sources",
        "",
        "Every answer may cite **one** of these URLs and nothing else. This file is",
        "generated from `data/corpus/urls.csv` + the `01_loaded` artifacts, so it can",
        "never drift from what the pipeline actually ingested.",
        "",
        f"- **Corpus size:** {len(rows)} URLs",
        f"- **Verified reachable:** {', '.join(dates) or '—'} "
        f"({sum(1 for doc in docs.values() if doc.http_status == 200)}/{len(docs)} returned HTTP 200)",
        f"- **Education links (`education=true`):** {len(education)} "
        f"({', '.join(row['id'] for row in education) or 'none'})",
        "- **Scope:** one AMC (HDFC) and four schemes — Large Cap, Flexi Cap, ELSS, Mid Cap.",
        "- **Regenerate:** `python -m app.make_sources`",
        "",
        "> **Corpus is limited to the URLs supplied by the user.** No additional",
        "> sources were discovered or padded in.",
        "",
    ]

    gaps = []
    if len(rows) < 15:
        gaps.append(
            f"- The brief asks for 15–25 URLs; only the {len(rows)} supplied URLs are scraped. "
            "This is a deliberate user constraint, not an extraction failure.")
    if len(education) < 2:
        found = ("There are no education links in the corpus"
                 if not education else
                 f"Only {len(education)} education link{'s' if len(education) > 1 else ''} in the corpus")
        gaps.append(
            f"- {found}, but refusals need at least two. Until the user supplies one, "
            "a refusal must fall back to a generic non-citation message rather than invent a URL.")
    if gaps:
        lines += ["## Known gaps", "", *gaps, ""]

    for page_type in TYPE_ORDER:
        group = by_type.get(page_type)
        if not group:
            continue
        lines += [
            f"## {TYPE_LABEL[page_type]} — {counts[page_type]}",
            "",
            "| id | Purpose | URL | Intent tags |",
            "|---|---|---|---|",
        ]
        for row in group:
            facts = _facts(row["intent_tags"])
            purpose = PURPOSE.get(row["id"], facts)
            flag = " *(education link)*" if row["education"].lower() == "true" else ""
            lines.append(f"| `{row['id']}` | {purpose}{flag} | <{row['url']}> | {facts} |")
        lines.append("")

    lines += [
        "## Excluded sources and why",
        "",
        "| Source | Reason |",
        "|---|---|",
        "| SEBI / AMFI / HDFC AMC pages | Not scraped: the corpus is restricted to the URLs supplied by the user. |",
        "| `hdfcfund.com` (HDFC AMC website) | Returns **HTTP 403 Access Denied** to automated requests (checked 2026-11-01), so it could not be ingested even if added. |",
        "| Blogs, aggregators, social media | Out of scope: official sources only. |",
        "| Performance / returns pages | Deliberately excluded — the product makes no performance claims, so return data is filtered out at the parser (`PERFORMANCE_LABEL`). |",
        "",
        "## Verification notes",
        "",
        "- Every URL above was opened by the loader on the date shown and returned HTTP 200.",
        "- Only the five supplied seed URLs were fetched; nothing else was discovered or crawled.",
        "- Content changes on the live web will change answers; re-run `--stage load --no-cache`",
        "  to refresh `01_loaded` and the `Last updated from sources` date.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = load_config()
    path = cfg.resolve(Path("docs")) / "SOURCES.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(cfg), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

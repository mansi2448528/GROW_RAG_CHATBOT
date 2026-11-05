"""HTML -> structural Blocks (implementation.md §2.3).

Two strategies, chosen by page_type:

  groww_scheme  -> read the page's own `__NEXT_DATA__` payload (server-rendered,
                   public, same URL) for exact facts, then fall back to DOM text
                   for anything missing. Facts arrive as labelled, atomic blocks.
  other pages   -> generic DOM extraction: tables (self-contained label: value
                   rows), definitions, headings and paragraphs.

One generic extractor + per-page_type hints. Never a per-URL patch.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from bs4 import BeautifulSoup, Tag

from src.models import Block, DocMeta

# CSS hints per page_type. Calibrated against the live pages on 2026-11-01.
# skip_table_headers: exact header-cell names that mark a table as out of scope
#   (performance comparisons, other schemes) — ingested returns would break the
#   "no performance claims" rule before guardrails even run.
# exclude_sections: section paths that are site navigation, not content.
SELECTORS: dict[str, dict[str, Any]] = {
    "groww_scheme": {"main": "main, #root, body", "max_table_rows": 20},
    "groww_amc": {
        "main": "main, #root, body", "max_table_rows": 20,
        "skip_table_headers": ["Fund Name", "1Y Returns", "Min Investment Amt"],
        "exclude_sections": ["AMC/Fund Houses"],
        # The AMC overview also markets every HDFC scheme. Per-scheme minimums for
        # funds outside the 4-scheme scope would be retrievable but unanswerable.
        "skip_paragraph_prefixes": ["Minimum Investment Amount"],
        "max_paragraphs": 12,
    },
    "amc_page": {"main": "main, article, .content", "max_table_rows": 20},
    "sebi_page": {"main": "main, article, #content", "max_table_rows": 20},
    "amfi_page": {"main": "main, article, #content", "max_table_rows": 20},
}

NOISE = re.compile(
    r"(sign ?in|log ?in|download app|invest now|start sip|blog|explore funds|"
    r"cookie|privacy policy|terms of use|all rights reserved)", re.I)

FACT_LABELS = {
    "expense_ratio": "Expense ratio",
    "exit_load": "Exit load",
    "min_sip": "Minimum SIP",
    "lock_in": "Lock-in",
    "riskometer": "Riskometer",
    "benchmark": "Benchmark",
    "statement_download": "Statement download",
}

RISK_LABEL = re.compile(r"\b(Very High|Moderately High|High|Moderate|Moderately Low|Low)\s+Risk\b", re.I)

# Rows whose label is a return figure are never ingested: the product must not
# answer performance questions at all, so the data should not exist to retrieve.
PERFORMANCE_LABEL = re.compile(r"^(1y|3y|5y|7y|10y|since inception)?\s*(returns?|cagr|xirr)$", re.I)

# Prose can smuggle performance claims in too ("Fund Performance: … 15.54%").
# Matched narrowly so benchmark names ("NIFTY 500 Total Return Index") survive.
PERFORMANCE_PROSE = re.compile(
    r"\b(fund performance|annuali[sz]ed returns?|returns? (?:for|over) the (?:past|last)|"
    r"\d+\s*[- ]?year returns?|cagr|xirr|since inception returns?)\b", re.I)


# ---------------------------------------------------------------- helpers

def _text(node: Tag | None) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip() if node else ""


def _inr(value: Any) -> str:
    """14948.74 -> '14,948.74' (Indian digit grouping)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    whole, _, frac = f"{number:.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    frac = frac.rstrip("0")
    return f"{whole}.{frac}" if frac else whole


def _next_data(html: str) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return None
    try:
        return json.loads(tag.string)
    except json.JSONDecodeError:
        return None


def _visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return _text(soup.body or soup)


def _dedupe(blocks: Iterable[Block]) -> list[Block]:
    seen: set[str] = set()
    out: list[Block] = []
    for block in blocks:
        key = re.sub(r"[^a-z0-9]+", "", block.text.lower())[:120]
        if len(block.text) < 3 or key in seen:
            continue
        seen.add(key)
        out.append(block)
    return out


# ------------------------------------------------- scheme page (JSON first)

def _scheme_json_blocks(html: str, meta: DocMeta) -> tuple[list[Block], dict[str, Any]]:
    payload = _next_data(html) or {}
    mf = (payload.get("props", {}).get("pageProps", {}) or {}).get("mfServerSideData") or {}
    if not mf:
        return [], {}

    blocks: list[Block] = []
    facts: dict[str, str] = {}

    def add(label: str, value: Any, fact_key: str | None = None) -> None:
        text = f"{label}: {value}"
        if fact_key:
            facts[fact_key] = text
        blocks.append(Block(["Fund facts"], "definition", text, fact_key=fact_key))

    def present(value: Any) -> bool:
        return value not in (None, "", "0", 0, [], {}, False)

    if present(mf.get("scheme_name")):
        add("Scheme name", mf["scheme_name"])
    if present(mf.get("plan_type")):
        add("Plan", str(mf["plan_type"]).replace("_", " ").title())
    if present(mf.get("category")):
        sub = mf.get("sub_category")
        add("Category", f"{mf['category']} – {sub}" if sub else mf["category"])

    if present(mf.get("expense_ratio")):
        add(FACT_LABELS["expense_ratio"], f"{mf['expense_ratio']}% per annum", "expense_ratio")
    if present(mf.get("base_expense_ratio")):
        add("Base expense ratio (excluding additional charges)", f"{mf['base_expense_ratio']}%")
    if present(mf.get("exit_load")):
        add(FACT_LABELS["exit_load"], mf["exit_load"], "exit_load")
    if present(mf.get("min_sip_investment")):
        add(FACT_LABELS["min_sip"], f"₹{_inr(mf['min_sip_investment'])}", "min_sip")
    if present(mf.get("min_investment_amount")):
        add("Minimum lump-sum investment", f"₹{_inr(mf['min_investment_amount'])}")

    lock_in = (mf.get("additional_details") or {}).get("lock_in_yrs")
    if present(lock_in) and float(lock_in) > 0:
        add(FACT_LABELS["lock_in"],
            f"{int(lock_in)} years from the date of investment", "lock_in")

    if present(mf.get("benchmark_name")) or present(mf.get("benchmark")):
        name, code = mf.get("benchmark_name"), mf.get("benchmark")
        value = f"{name} ({code})" if name and code else (name or code)
        add(FACT_LABELS["benchmark"], value, "benchmark")

    if present(mf.get("aum")):
        add("Fund size (AUM)", f"₹{_inr(mf['aum'])} crore")
    if present(mf.get("nav")):
        when = f" as on {mf['nav_date']}" if mf.get("nav_date") else ""
        add("NAV", f"₹{_inr(mf['nav'])}{when}")
    if present(mf.get("fund_house")) or present(mf.get("amc_info", {}).get("name")):
        add("Fund house", mf.get("fund_house") or mf.get("amc_info", {}).get("name"))
    if present(mf.get("registrar_agent")):
        add("Registrar (RTA)", mf["registrar_agent"])
    if present(mf.get("launch_date")):
        add("Launch date", mf["launch_date"])

    managers = mf.get("fund_manager_details") or mf.get("fund_manager")
    if isinstance(managers, list) and managers:
        names = [m.get("name") if isinstance(m, dict) else str(m) for m in managers]
        names = [n for n in names if n]
        if names:
            add("Fund manager", ", ".join(names))
    elif isinstance(managers, str) and managers:
        add("Fund manager", managers)

    # Riskometer is not a JSON field: it is rendered text ("Very High Risk").
    match = RISK_LABEL.search(_visible_text(html))
    if match:
        label = f"{match.group(1).title()} Risk"
        add(FACT_LABELS["riskometer"], label, "riskometer")

    description = mf.get("description")
    if isinstance(description, str) and len(description) > 80:
        blocks.append(Block(["About the fund"], "paragraph",
                            re.sub(r"\s+", " ", description).strip()))

    return blocks, facts


def _missing_facts(blocks: list[Block]) -> list[str]:
    found = {b.fact_key for b in blocks if b.fact_key}
    return [key for key in FACT_LABELS if key not in found and key not in ("statement_download",)]


# ------------------------------------------------ generic page (DOM blocks)

def _table_blocks(table: Tag, max_rows: int, skip_headers: list[str] | None = None) -> list[Block]:
    rows = table.find_all("tr")
    if not rows:
        return []
    header_cells = rows[0].find_all(["th"])
    first_cells = [_text(c) for c in rows[0].find_all(["th", "td"])]

    if skip_headers:
        skip = {s.casefold() for s in skip_headers}
        if any(cell.casefold() in skip for cell in first_cells):
            return []                      # out of scope: performance / other schemes

    has_header = bool(header_cells) and len(header_cells) == len(first_cells)

    blocks: list[Block] = []
    if has_header:
        header = first_cells
        data_rows = rows[1:]
    else:
        header, data_rows = [], rows

    if len(data_rows) > max_rows:
        data_rows = data_rows[:max_rows]

    for row in data_rows:
        cells = [_text(c) for c in row.find_all(["th", "td"])]
        if len(cells) < 2:
            continue
        if header:
            pairs = [f"{h}: {v}" for h, v in zip(header, cells) if v]
        else:
            pairs = [f"{cells[0]}: {cells[1]}"] + cells[2:]
        text = " | ".join(pairs)
        if len(text) < 8:
            continue
        if any(PERFORMANCE_LABEL.match(pair.split(":", 1)[0].strip()) for pair in pairs):
            continue                       # performance row — out of scope by design
        blocks.append(Block(["Table"], "table_row", text, rows=[header or cells[:1], cells]))
    return blocks


def _generic_blocks(html: str, meta: DocMeta) -> list[Block]:
    hints = SELECTORS.get(meta.page_type, SELECTORS["amfi_page"])
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()
    root = soup.select_one(hints["main"]) or soup.body or soup

    skip_headers = hints.get("skip_table_headers", [])
    skip_prefixes = tuple(p.lower() for p in hints.get("skip_paragraph_prefixes", []))
    max_paragraphs = int(hints.get("max_paragraphs", 40))
    blocks: list[Block] = []
    path: list[str] = []
    paragraphs = 0

    def skipped(text: str) -> bool:
        return bool(skip_prefixes) and text.lower().startswith(skip_prefixes)

    for el in root.find_all(["h1", "h2", "h3", "h4", "table", "li", "p"]):
        name = el.name
        if name in ("h1", "h2", "h3", "h4"):
            heading = _text(el)
            if not heading:
                continue
            level = int(name[1])
            path = path[: level - 1] + [heading]
        elif name == "table":
            blocks.extend(_table_blocks(el, hints["max_table_rows"], skip_headers))
        elif name == "li":
            text = _text(el)
            if (40 < len(text) < 600 and not NOISE.search(text)
                    and not PERFORMANCE_PROSE.search(text) and not skipped(text)):
                blocks.append(Block(path or ["Page"], "list_item", text))
        else:
            text = _text(el)
            if (80 < len(text) < 2000 and not NOISE.search(text) and paragraphs < max_paragraphs
                    and not PERFORMANCE_PROSE.search(text) and not skipped(text)):
                blocks.append(Block(path or ["Page"], "paragraph", text))
                paragraphs += 1

    excluded = hints.get("exclude_sections", [])
    if excluded:
        blocks = [b for b in blocks
                  if not any(" > ".join(b.section_path).startswith(prefix) for prefix in excluded)]
    return blocks


# ------------------------------------------------------------------- public

def parse(html: str, meta: DocMeta) -> tuple[list[Block], list[str]]:
    """Return (blocks, warnings) for one document."""
    warnings: list[str] = []
    if not html:
        return [], ["empty html"]

    if meta.page_type == "groww_scheme":
        blocks, facts = _scheme_json_blocks(html, meta)
        if not blocks:
            warnings.append("no __NEXT_DATA__ on scheme page — fell back to DOM text")
        else:
            missing = [k for k in _missing_facts(blocks) if k != "lock_in" or meta.scheme == "hdfc_elss"]
            if missing:
                warnings.append(f"facts not found: {', '.join(missing)}")
            for key in missing:                      # last resort: scan visible text
                fallback = _fact_from_text(html, key)
                if fallback:
                    blocks.append(Block(["Fund facts (page text)"], "definition",
                                        fallback, fact_key=key))
        blocks.extend(_generic_prose(html, meta))
        return _dedupe(blocks), warnings

    blocks = _generic_blocks(html, meta)
    if not blocks:
        warnings.append("no blocks extracted")
    return _dedupe(blocks), warnings


def _generic_prose(html: str, meta: DocMeta) -> list[Block]:
    """Paragraphs only — used on scheme pages, where facts come from JSON."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()
    out: list[Block] = []
    for el in soup.find_all("p"):
        text = _text(el)
        if 100 < len(text) < 1200 and not NOISE.search(text) and not PERFORMANCE_PROSE.search(text):
            out.append(Block(["Page text"], "paragraph", text))
    return out[:8]


def _fact_from_text(html: str, fact_key: str) -> str | None:
    text = _visible_text(html)
    label = FACT_LABELS[fact_key]
    match = re.search(re.escape(label) + r"\s*[:\-–]?\s*([^|]{1,120})", text, re.I)
    if not match:
        return None
    value = match.group(1).strip(" .;:-")
    return f"{label}: {value}" if value else None

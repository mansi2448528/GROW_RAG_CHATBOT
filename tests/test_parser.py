"""Parser + manifest tests (implementation.md §2). No network needed."""

from __future__ import annotations

import json

import pytest

from config.config import ConfigError
from src.loader import validate_manifest
from src.models import DocMeta
from src.parser import parse

MF_HTML = """<html><head><title>HDFC ELSS Tax Saver Fund Direct Plan Growth</title></head>
<body><p>Very High Risk</p><script id="__NEXT_DATA__" type="application/json">
{"props":{"pageProps":{"mfServerSideData":{
  "scheme_name":"HDFC ELSS Tax Saver Fund Direct Plan Growth",
  "plan_type":"DIRECT","category":"Equity","sub_category":"ELSS",
  "expense_ratio":1.2,"base_expense_ratio":0.97,"exit_load":"Nil",
  "min_sip_investment":500,"min_investment_amount":500,
  "additional_details":{"lock_in_yrs":3},
  "benchmark":"NIFTY 500 TRI","benchmark_name":"NIFTY 500 Total Return Index",
  "aum":14948.74,"nav":1434.915,"nav_date":"30-Oct-2026",
  "fund_house":"HDFC Mutual Fund","registrar_agent":"Cams",
  "description":"The scheme seeks to generate capital appreciation from a portfolio comprising predominantly equity and equity related instruments across market capitalisations and sectors."
}}}}</script></body></html>"""

TABLE_HTML = """<html><body><main>
<h2>Returns</h2>
<table>
  <tr><th>Fund Name</th><th>1Y Returns</th><th>Exit Load</th></tr>
  <tr><td>HDFC Mid Cap Fund</td><td>-2.5%</td><td>1% within 1 year</td></tr>
</table>
<table>
  <tr><td>AUM</td><td>Rs 1,01,856 Cr</td></tr>
  <tr><td>Minimum Investment</td><td>Rs 100</td></tr>
</table>
</main></body></html>"""


def scheme_meta(scheme: str = "hdfc_elss") -> DocMeta:
    return DocMeta(doc_id=f"groww_{scheme}", source_url="https://groww.in/x",
                   page_title="", page_type="groww_scheme", scheme=scheme,
                   amc="HDFC Mutual Fund", plan="direct_growth")


def test_scheme_page_yields_all_facts():
    blocks, warnings = parse(MF_HTML, scheme_meta())
    facts = {b.fact_key: b.text for b in blocks if b.fact_key}

    assert "expense_ratio" in facts and "1.2%" in facts["expense_ratio"]
    assert facts["exit_load"].endswith("Nil")
    assert "500" in facts["min_sip"]
    assert "3 years" in facts["lock_in"]
    assert "NIFTY 500 Total Return Index" in facts["benchmark"]
    assert facts["riskometer"] == "Riskometer: Very High Risk"
    assert warnings == []


def test_lock_in_absent_for_non_elss():
    html = MF_HTML.replace('"lock_in_yrs":3', '"lock_in_yrs":0')
    blocks, _ = parse(html, scheme_meta("hdfc_mid_cap"))
    assert not any(b.fact_key == "lock_in" for b in blocks)


def test_performance_tables_are_not_ingested():
    meta = scheme_meta("hdfc_mid_cap")
    meta.page_type = "groww_amc"
    blocks, _ = parse(TABLE_HTML, meta)
    text = " ".join(b.text for b in blocks)

    assert "Returns" not in text          # scheme list with return columns skipped
    assert "AUM: Rs 1,01,856 Cr" in text  # label/value table kept
    assert not any(b.kind == "table_row" and "Fund Name" in b.text for b in blocks)


def test_empty_html_is_reported_not_silent():
    blocks, warnings = parse("", scheme_meta())
    assert blocks == [] and warnings == ["empty html"]


def test_performance_prose_is_not_ingested():
    html = """<html><body><main>
    <p>Fund Performance: The fund's annualized returns for the past 3 years and 5 years
    have been around 15.54% and 17.07% respectively, which is above the category average.</p>
    <p>Expense ratio of the direct plan is 1.2% per annum, charged daily on the net asset
    value of the scheme, and disclosed in the scheme information document.</p>
    </main></body></html>"""
    meta = scheme_meta("hdfc_mid_cap")
    meta.page_type = "groww_amc"
    blocks, _ = parse(html, meta)
    text = " ".join(b.text for b in blocks)

    assert "annualized returns" not in text and "15.54" not in text
    assert "Expense ratio" in text


def test_amc_unattributed_scheme_cards_are_not_ingested():
    # Groww renders per-scheme side cards whose fund name sits outside the table:
    # "AUM: ₹1,01,856Cr" cannot be attributed to a scheme, so it must not become
    # a retrievable fact (the scheme pages carry the same numbers, with a name).
    html = """<html><body><main>
    <table><tr><td>Min Investment Amt</td><td>₹100</td></tr>
    <tr><td>AUM</td><td>₹1,01,856Cr</td></tr>
    <tr><td>1Y Returns</td><td>-2.5%</td></tr></table>
    </main></body></html>"""
    meta = scheme_meta("hdfc_mid_cap")
    meta.page_type = "groww_amc"
    blocks, _ = parse(html, meta)

    text = " ".join(b.text for b in blocks)
    assert "1,01,856" not in text and "Min Investment Amt" not in text


def test_amc_per_scheme_marketing_is_not_ingested():
    html = """<html><body><main>
    <p>Minimum Investment Amount: Lump sum minimum amount for HDFC Silver ETF FoF
    Direct-Growth is Rs 100 and for SIP, it is Rs 100 as well.</p>
    </main></body></html>"""
    meta = scheme_meta("hdfc_mid_cap")
    meta.page_type = "groww_amc"
    blocks, _ = parse(html, meta)

    assert not any("Silver ETF" in b.text for b in blocks)


def test_manifest_rejects_duplicates_and_bad_page_type():
    base = {"id": "a", "url": "https://example.com/a", "page_type": "sebi_page",
            "scheme": "", "amc": "SEBI", "plan": "na", "intent_tags": "", "education": "true"}
    with pytest.raises(ConfigError, match="duplicate id"):
        validate_manifest([base, dict(base)])

    bad = dict(base, id="b", url="https://example.com/b", page_type="blog")
    with pytest.raises(ConfigError, match="unknown page_type"):
        validate_manifest([base, bad])


def test_manifest_counts_are_notes_not_errors():
    row = {"id": "a", "url": "https://example.com/a", "page_type": "sebi_page",
           "scheme": "", "amc": "SEBI", "plan": "na", "intent_tags": "", "education": "false"}
    notes = validate_manifest([row])
    assert any("corpus incomplete" in n for n in notes)
    assert any("education=true" in n for n in notes)

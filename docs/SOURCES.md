# Sources

Every answer may cite **one** of these URLs and nothing else. This file is
generated from `data/corpus/urls.csv` + the `01_loaded` artifacts, so it can
never drift from what the pipeline actually ingested.

- **Corpus size:** 5 URLs
- **Verified reachable:** 2026-11-01 (5/5 returned HTTP 200)
- **Education links (`education=true`):** 0 (none)
- **Scope:** one AMC (HDFC) and four schemes — Large Cap, Flexi Cap, ELSS, Mid Cap.
- **Regenerate:** `python -m app.make_sources`

> **Corpus is limited to the URLs supplied by the user.** No additional
> sources were discovered or padded in.

## Known gaps

- The brief asks for 15–25 URLs; only the 5 supplied URLs are scraped. This is a deliberate user constraint, not an extraction failure.
- There are no education links in the corpus, but refusals need at least two. Until the user supplies one, a refusal must fall back to a generic non-citation message rather than invent a URL.

## Scheme pages (Groww) — 4

| id | Purpose | URL | Intent tags |
|---|---|---|---|
| `groww_hdfc_large_cap` | Facts for HDFC Large Cap Fund (Direct-Growth) | <https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth> | expense ratio, exit load, minimum SIP, riskometer, benchmark, fund basics |
| `groww_hdfc_flexi_cap` | Facts for HDFC Flexi Cap Fund (Direct-Growth) | <https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth> | expense ratio, exit load, minimum SIP, riskometer, benchmark, fund basics |
| `groww_hdfc_elss` | Facts for HDFC ELSS Tax Saver (Direct-Growth), incl. 3-year lock-in | <https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth> | expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark |
| `groww_hdfc_mid_cap` | Facts for HDFC Mid Cap Fund (Direct-Growth) | <https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth> | expense ratio, exit load, minimum SIP, riskometer, benchmark, fund basics |

## AMC overview (Groww) — 1

| id | Purpose | URL | Intent tags |
|---|---|---|---|
| `groww_hdfc_amc` | Who runs HDFC Mutual Fund, trustees, sponsor, contact | <https://groww.in/mutual-funds/amc/hdfc-mutual-funds> | AMC overview |

## Excluded sources and why

| Source | Reason |
|---|---|
| SEBI / AMFI / HDFC AMC pages | Not scraped: the corpus is restricted to the URLs supplied by the user. |
| `hdfcfund.com` (HDFC AMC website) | Returns **HTTP 403 Access Denied** to automated requests (checked 2026-11-01), so it could not be ingested even if added. |
| Blogs, aggregators, social media | Out of scope: official sources only. |
| Performance / returns pages | Deliberately excluded — the product makes no performance claims, so return data is filtered out at the parser (`PERFORMANCE_LABEL`). |

## Verification notes

- Every URL above was opened by the loader on the date shown and returned HTTP 200.
- Only the five supplied seed URLs were fetched; nothing else was discovered or crawled.
- Content changes on the live web will change answers; re-run `--stage load --no-cache`
  to refresh `01_loaded` and the `Last updated from sources` date.

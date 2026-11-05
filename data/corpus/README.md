# Corpus manifest

`urls.csv` is the **only** source of citations. A URL that is not in this file can never
appear in an answer (validator V4, implementation.md §8.2).

## Current state

| page_type | rows | notes |
|---|---|---|
| `groww_scheme` | 4 | Large Cap, Flexi Cap, ELSS, Mid Cap — the mandatory seed pages |
| `groww_amc` | 1 | HDFC Mutual Fund overview |
| `amc_page` | 0 | not scraped — corpus is limited to the URLs supplied by the user |
| `sebi_page` | 0 | not scraped — corpus is limited to the URLs supplied by the user |
| `amfi_page` | 0 | not scraped — corpus is limited to the URLs supplied by the user |
| **total** | **5** | the 5 supplied seed URLs; the brief's 15–25 target is intentionally not padded |

Two known gaps follow from this scope (tracked in `docs/SOURCES.md`):

1. The brief asks for 15–25 URLs; only the 5 supplied URLs are scraped.
2. There are **no** `education=true` pages, so the refusal path has no official
   education link yet. Do not invent one — add a URL the user supplies.

`docs/SOURCES.md` is generated from this file by `python -m app.make_sources`; edit the
manifest, not the docs.

## Rules

1. Public, official pages only: Groww product pages, HDFC AMC, SEBI, AMFI. No blogs,
   no aggregators, no screenshots of app back-ends.
2. Every row must be opened manually once and confirmed reachable — record the
   verification date in `docs/SOURCES.md`.
3. `education: true` marks pages that may be used as the "read more" link in a
   refusal. Need at least two.
4. `intent_tags` uses the vocabulary in `config/lexicon.yaml` (`fact_keys`) so the
   fact-key filter in §6 has something to match.
5. `scheme` must be one of: `hdfc_large_cap`, `hdfc_flexi_cap`, `hdfc_elss`,
   `hdfc_mid_cap` (or empty for AMC/regulator pages).

`python -m app.pipeline --check-config` reports the count and what is missing.

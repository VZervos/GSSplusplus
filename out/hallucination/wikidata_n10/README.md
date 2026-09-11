# n=10 Wikidata GSS rerun

Same archived LLM answers as the DBpedia n=10 pilot (`llm_answers.json`), with `DATASET=wikidata` (SPARQL `https://query.wikidata.org/sparql`).

**Wall time:** ~63 minutes (3 modes × 10).  
**Code tweaks for Wikidata (outside GSS core algorithm):** Q-ID URI handling, wbsearchentities label lookup, allowlist P-ids, SPARQL User-Agent / 403–429 retries, inter-item sleep.

## Summary vs DBpedia n=10

| Metric | WD answer | WD subject | WD claim | DBP answer | DBP subject | DBP claim |
|--------|-----------|------------|----------|------------|-------------|-----------|
| supported | 0 | 0 | 0 | 1 | 3 | 3 |
| weak_support | 0 | 0 | 0 | 1 | 1 | 0 |
| unsupported | 4 | 4 | 8 | 8 | 5 | 6 |
| uncertain | 6 | 6 | 2 | 0 | 1 | 1 |
| true_support | 0 | 0 | 0 | 1 | 2 | 2 |
| false_support | 0 | 0 | 0 | 0 | 0 | 0 |
| missed_support | 1 | 2 | 3 | 3 | 1 | 2 |
| gold_match | 4 | 4 | 4 | 4 | 4 | 4 |

## Takeaway

On this 10-item set, **Wikidata GSS did not improve** claim validation over DBpedia: **zero strict supports**, more **empty ACEs** (endpoint 502/504/rate-limit pressure), and several gold-correct answers left as `missed_support` even with non-empty subgraphs (e.g. Tirana/Geneva/London under claim). False support stays 0.

Artifacts: `validation_results_{answer,subject,claim}.json`, `run_wikidata_n10.log`.

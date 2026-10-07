# CoLoTa demo — iSummary official JAR, Wikidata query log (n=20)

**Date:** 2026-10-06  
**Live code:** `src/colota_demo/approaches/isummary/` (JAR wrapper only)  
**System:** `third_party/iSummary/isummary.jar` (Vassiliou, Papadakis, Kondylakis, SWJ 2025)  
**Workload:** authors’ Wikidata `train.txt` (`third_party/iSummary/data/wikidata/`)  
**Raw outputs:** `results.csv`, `results.json`, `metrics.json`, `isummary_cache/`, `isummary_work/` in this folder  
**Previous runs:** [`../first_experiment/REPORT.md`](../first_experiment/REPORT.md) (LLM / gold / MiniLM RAG), [`../gss/REPORT.md`](../gss/REPORT.md) (live Wikidata GSS), [`../gss_corpus/REPORT.md`](../gss_corpus/REPORT.md) (GSS on the 37-triple corpus)

Same 20 hand-picked CoLoTa QA items. iSummary-only. The summarizer is the **official JAR** on the paper’s **Wikidata SPARQL log**, not a Python reimplementation and not live Wikidata SPARQL.

---

## 1. Motivation

Earlier conditions asked whether *question-specific KG context* helps llama3.1:8b:

| Condition | Accuracy@5 |
|-----------|------------|
| LLM-only | 50% |
| Gold triples | 90% |
| MiniLM RAG over 37 gold triples | 70% |
| GSS live Wikidata | 50% |
| GSS on the same 37 triples | 40% |

This run uses **iSummary** as published: a (λ, κ)-selective summary from a **query log**, not a 1-hop walk on the KG. CoLoTa is Wikidata-grounded, so the authors’ Wikidata dump (`train.txt` / `test.txt` / `nodes2.txt`) is the matching workload.

---

## 2. Setup

### 2.1 Official system

- **JAR:** `java -jar isummary.jar <testdata> <traindata> <nodes> <top_k> <choose_from>`
- **Train log:** `third_party/iSummary/data/wikidata/train.txt` (~154k queries)
- **Test file:** dummy one-query TSV. Testdata is only used for the JAR’s coverage numbers, which we do not report. Summary construction uses the train log.
- **top_k / kappa:** 12 (JAR size gate: abort if the weighted node set is smaller than `4 × 12 / 5 = 9`)
- **choose_from:** 2, with a 2-line ranking file that puts the CoLoTa seed first. The JAR’s `nextInt(1, choose_from)` cannot be 1; this is the only seed adapter.
- **Variable resolution:** log-only (no SPARQL endpoint)

Python does not run Algorithm 1. It maps CoLoTa `kg_entities` to `http://www.wikidata.org/entity/Q…`, writes the ranking file, invokes the JAR, and parses `---------- PARTI` paths.

### 2.2 Seeds and empty policy

- Seed = first QID in `kg_entities` whose URI appears in the train log.
- If none appear: skip the JAR, empty subgraph.
- Empty subgraph → answering `INSUFFICIENT`, Recall@k = 0.

### 2.3 Answering and metrics

- **LLM:** local Ollama `llama3.1:8b`, temperature 0
- **Prompt:** question + iSummary paths; JSON `TRUE` / `FALSE` / `INSUFFICIENT`
- **Depths:** k ∈ {1, 3, 5, 10}; primary k = 5
- **Ranking gold:** that item’s English `kg_triples` (exact string match), same helpers as RAG/GSS
- Hits@k = first relevant rank ≤ k; `INSUFFICIENT` counts as incorrect

### 2.4 Items

Identical `DEMO_IDS` (10 True / 10 False). Not a random CoLoTa sample.

### 2.5 Run notes

Wall time ~8.5 minutes. GSS-corpus was still using Ollama in parallel; iSummary queued on the same local model.

---

## 3. Results

### 3.1 Main conditions (k=5)

| Condition | Accuracy | Role |
|-----------|----------|------|
| LLM-only (first experiment) | **50.0%** | parametric baseline |
| Gold triples (first experiment) | **90.0%** | reasoning ceiling |
| RAG@5 (first experiment) | **70.0%** | MiniLM over the 37 triples |
| GSS@5 live Wikidata | **50.0%** | live 1-hop + NER |
| GSS@5 corpus | **40.0%** | GSS over the 37 triples |
| **iSummary@5 (this run)** | **10.0%** | official JAR + Wikidata log |

iSummary@5 is far below LLM-only. Almost every item abstains.

### 3.2 Depth

| k | iSummary accuracy | Recall | Hits@k |
|---|-------------------|--------|--------|
| 1 | 10.0% | 0.0% | 0% |
| 3 | 10.0% | 0.0% | 0% |
| 5 | **10.0%** | **0.0%** | **0%** |
| 10 | 10.0% | 0.0% | 0% |

MRR = 0, Mean RR = 0. No gold `kg_triples` string ever appears in an iSummary path. Depth does not change answers because summaries are empty.

### 3.3 What the log actually contained

| Seed in Wikidata `train.txt` | Items |
|------------------------------|--------|
| **0 mentions** | 18/20 |
| **Mentioned** | S51 Serie A `Q15804` (13 queries), S160 Tehran `Q3616` (1 query) |

JAR invocations:

- **S51:** 13 queries parsed. Weighted node set size 2 (`Q15804`, `rdf#serviceParam`). Original `BROKER FOR` (2 < 9). Cache `[]`.
- **S160:** 1 query matched, Jena parse success 0. Empty `hm`, retry same seed, break. Cache `[]`.

Berlin `Q64` (not in `DEMO_IDS`) does produce PARTI paths at kappa=3; that seed is frequent in the paper log. This CoLoTa slice is long-tail relative to that log.

### 3.4 Per-item

See `results.csv`. `INSUFFICIENT` counts as incorrect.

| ID | Gold | iSummary@5 | R@5 | Notes |
|----|------|------------|-----|-------|
| S1 | True | INSUFFICIENT | 0% | Horsens/Ikast absent from log |
| S2 | False | INSUFFICIENT | 0% | |
| S3 | True | INSUFFICIENT | 0% | |
| S5 | True | INSUFFICIENT | 0% | |
| S13 | False | INSUFFICIENT | 0% | |
| S37 | False | INSUFFICIENT | 0% | |
| S51 | True | INSUFFICIENT | 0% | JAR `BROKER FOR` Q15804 |
| S87 | False | INSUFFICIENT | 0% | |
| S95 | False | **False** | 0% | Empty graph; model said False |
| S131 | False | INSUFFICIENT | 0% | |
| S142 | True | INSUFFICIENT | 0% | |
| S160 | True | INSUFFICIENT | 0% | JAR ran; 1 log hit, 0 Jena-ok |
| S164 | True | INSUFFICIENT | 0% | |
| S175 | False | INSUFFICIENT | 0% | |
| S184 | False | INSUFFICIENT | 0% | |
| S4 | False | INSUFFICIENT | 0% | |
| S10 | False | **False** | 0% | Empty graph; model said False |
| S34 | True | INSUFFICIENT | 0% | |
| S55 | True | INSUFFICIENT | 0% | |
| S124 | True | INSUFFICIENT | 0% | |

**Correct at k=5:** S95, S10 only (2/20), both Gold-False guesses on empty context.  
**True items correct:** 0/10.  
**INSUFFICIENT at k=5:** 18/20.

---

## 4. Findings

1. **Official iSummary on the official Wikidata workload does not cover this CoLoTa slice.** 18/20 QIDs never appear in `train.txt`. The paper log is popularity-biased (Q5, Q11424, Q30, …), not long-tail CoLoTa entities.
2. **Even the two log hits produce no summary at kappa=12.** S51 is aborted by the original size gate. S160’s single query does not Jena-parse. Empty subgraph is the measured system, not a wrapper bug.
3. **10% accuracy is not evidence of useful context.** Both hits are Gold-False + empty → False, the same pattern as empty GSS on S95/S10. Ranking vs English gold triples is identically 0 (URI/path vs `(Horsens, population, …)`).
4. **This is a workload mismatch, not a JAR failure.** The same binary builds a Berlin (`Q64`) summary when the seed is frequent in the log.

### Limitations

- n=20, one local 8B model.
- Testdata is a dummy file (coverage unused).
- Seed is injected via a 2-line ranking file instead of random `nodes2.txt` top-20.
- CoLoTa gold triples are English strings; iSummary outputs SPARQL paths, so Recall@k/MRR are expected to be 0 even on a non-empty summary.
- kappa=12 is the configured paper-style size; a smaller top_k would change the BROKER gate (S51 still had only 2 weighted nodes).

---

## 5. How to reproduce

From the repo root, with Ollama serving `llama3.1:8b` and the Wikidata dump under `third_party/iSummary/data/wikidata/`:

```text
python src/colota_demo/run_suite.py --approaches isummary --experiment isummary
```

`config.json` → `isummary.kappa = 12`, `isummary.train = third_party/iSummary/data/wikidata/train.txt`. Caches: `isummary_cache/{id}_k12.json`. JAR traces: `isummary_work/{id}/stdout.txt`.

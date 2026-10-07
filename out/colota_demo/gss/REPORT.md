# CoLoTa demo — archived GSS-only, live Wikidata (n=20)

**Date:** 2026-10-05 (archived 2026-10-06)  
**Code snapshot:** `code/` in this folder  
**Live code (corpus GSS, later):** `src/colota_demo/`  
**Raw outputs:** `results.csv`, `results.json`, `metrics.json`, `gss_cache/` in this folder  
**Sister archive (LLM / gold triples / MiniLM RAG):** [`../first_experiment/REPORT.md`](../first_experiment/REPORT.md)  
**Later iSummary (official JAR, Wikidata log):** [`../isummary/REPORT.md`](../isummary/REPORT.md)

Same 20 hand-picked CoLoTa QA items as the first experiment. Only GSS ran here; LLM-only, gold triples, and RAG numbers below are copied from that archive for comparison.

---

## 1. Motivation

The first experiment asked whether *question-specific KG context* helps llama3.1:8b on long-tail yes/no questions. Oracle `kg_triples` reached 90%; MiniLM RAG over a 37-triple pooled corpus sat at 70% at k=5.

This run asks the same answering question with a **live GSS subgraph** instead of a closed corpus:

- NER on the question (same local `llama3.1:8b`)
- Wikidata URI lookup + SPARQL 1-hop
- prune → importance / similarity → rank with paper mixing weight **a = 0.75**
- then a **separate** answering prompt at k ∈ {1, 3, 5, 10}, same protocol as RAG (may abstain with `INSUFFICIENT`)

GSS ranking / SPARQL / scoring were not changed for this experiment. NER JSON peel / `//` comment strip is glue only, so llama3.1:8b’s wrapped replies still parse.

---

## 2. Setup

### 2.1 Model and GSS

- **LLM:** local Ollama `llama3.1:8b`, temperature 0
- **NER:** original GSS prompt via Ollama; whole-reply JSON after fence / object peel / comment strip
- **KG:** live `query.wikidata.org` (retries on 504)
- **Rank:** `final_score = 0.75 × importance + 0.25 × similarity`
- **Answering:** question + top-k GSS triples; JSON `TRUE` / `FALSE` / `INSUFFICIENT`
- **Context format:** original triple fields (`s`, `p`, `o` URIs). `s_label` / `p_label` / `o_label` are empty in this pipeline, so the answering model mostly sees Wikidata URIs, not English labels.

### 2.2 Items

Identical to the first experiment (`DEMO_IDS` in `src/colota_demo/utils/dataset.py`): 10 True / 10 False, entity-aligned, non-empty gold `kg_triples`. Not a random CoLoTa sample.

### 2.3 Metrics

GSS-only, so the original demo metrics that need RAG rankings (Recall@k, MRR, failure tags vs LLM/gold/RAG) are empty. Reported here:

- Accuracy at k = 1, 3, 5, 10 (`INSUFFICIENT` = incorrect)
- Primary reporting k = 5, matching the first experiment

### 2.4 Run notes

Subgraphs were built in a ~2 hour live SPARQL pass (`gss_cache/{id}_a0.75.json`). The process died after S124’s subgraph was written, before `results.csv`. Answering was resumed from those caches (no SPARQL re-run). Per-item `gss@5` matched the first pass on all 19 items that had already printed.

---

## 3. Results

### 3.1 Main conditions (k=5)

| Condition | Accuracy | Role |
|-----------|----------|------|
| LLM-only (first experiment) | **50.0%** | parametric baseline |
| Gold triples (first experiment) | **90.0%** | reasoning ceiling |
| RAG@5 (first experiment) | **70.0%** | closed MiniLM corpus |
| **GSS@5 (this run)** | **50.0%** | live Wikidata subgraph |

GSS@5 ties LLM-only and is well below RAG@5 and the gold-triple ceiling. On this slice, live GSS context did not beat parametric memory.

### 3.2 Depth

| k | GSS accuracy |
|---|--------------|
| 1 | **55.0%** |
| 3 | 40.0% |
| 5 | 50.0% |
| 10 | 30.0% |

More triples make answers **worse**. k=1 is the best depth; k=10 is the worst. That matches the first experiment’s “bigger k is a worse summary” finding, but here accuracy falls *below* chance at k=10.

### 3.3 Per-item (GSS vs first-experiment RAG@5 / gold)

See `results.csv`. Gold True items are almost all wrong; most of the 50% comes from Gold-False items that GSS also labels False.

| ID | Gold | GSS@1 | GSS@3 | GSS@5 | GSS@10 | Notes |
|----|------|-------|-------|-------|--------|-------|
| S1 | True | INSUFFICIENT | False | False | INSUFFICIENT | Cached subgraph from smoke (Horsens/Ikast); still wrong |
| S2 | False | False | False | False | False | NER leaked film/GMT QIDs; default False happens to match |
| S3 | True | INSUFFICIENT | False | False | False | Leopold Lanner extracted; subgraph not enough |
| S5 | True | False | False | **True** | False | Few-shot GMT leak; k=5 is a lucky flip |
| S13 | False | False | False | False | True | Vaughan + Jamjuree mixed with GMT few-shot |
| S37 | False | False | True | True | True | Gujan/Aousserd mixed with GMT; false positive at k≥3 |
| S51 | True | False | False | **True** | False | GMT leak; another k=5-only hit |
| S87 | False | False | False | False | False | Paolo Cannavaro extracted |
| S95 | False | False | False | False | False | GMT few-shot leak |
| S131 | False | False | False | False | False | Pinar Abay + invented `Q1234567` |
| S142 | True | False | False | False | False | US-capital few-shot (Q30/P36) |
| S160 | True | INSUFFICIENT | INSUFFICIENT | INSUFFICIENT | INSUFFICIENT | Ranked subgraph size **0** |
| S164 | True | False | False | False | False | Film/director few-shot (Q11424/P57) |
| S175 | False | False | True | True | True | Lee Hyori + Disco extracted; false positive at k≥3 |
| S184 | False | False | False | False | INSUFFICIENT | Q11424/P57 leak |
| S4 | False | False | False | False | False | Q30/P36 leak |
| S10 | False | False | False | False | False | Q11424 leak |
| S34 | True | False | False | False | False | Dowlatabadi + *Makioka Sisters* extracted; still wrong |
| S55 | True | **True** | False | False | False | Q30/P36 leak; only correct at k=1 |
| S124 | True | False | False | False | False | Ivan Shuisky + Benny Frey extracted; URI triples, still wrong |

**Correct at k=5:** S2, S5, S13, S51, S87, S95, S131, S184, S4, S10 (10/20).  
**True items correct at k=5:** S5, S51 only (2/10).  
**INSUFFICIENT at k=5:** S160 only.

### 3.4 What the subgraphs actually contained

Typical pattern in the SPARQL log:

1. **Few-shot NER leakage.** The GSS NER prompt’s examples (GMT / board game, US capital Q30/P36, film Q11424/P57) are copied into many items that are about someone else. Those QIDs then dominate SPARQL (and often 504 on high-degree nodes like Q30).
2. **Empty ranked subgraph.** S160: Indonesia-style QIDs, prune/rank with `RANKING_MIN_SCORE = 0.1` left **0** triples → `INSUFFICIENT` at every k.
3. **URI-only answering context.** Even when NER hits the right people (S3, S34, S87, S124, S175), top-k lines look like `(http://www.wikidata.org/entity/Q…, http://www.wikidata.org/prop/…, http://www.wikidata.org/entity/statement/…)`. The answering model is not given the same readable facts as gold triples or MiniLM RAG.
4. **Wikidata 504s.** Expected on noisy high-degree QIDs; GSS retried and continued. Not a hang.

---

## 4. Findings

1. **GSS@5 = 50% = LLM-only** on this 20-item slice. Live 1-hop Wikidata did not reproduce the RAG@5 gain (70%) or the gold-triple ceiling (90%).
2. **Depth hurts.** 55% → 40% → 50% → 30% at k=1,3,5,10. Extra GSS triples are mostly distractors (statement nodes, leaked few-shot QIDs), not a cleaner query-focused summary.
3. **Most of the 50% is “Gold is False and the model says False.”** True items are almost unsolved (2/10 at k=5). That is not evidence the subgraph supported the commonsense step.
4. **The bottleneck is not “GSS ranking math.”** NER copies prompt entities, SPARQL follows those URIs, labels never reach the answering prompt, and min-score ranking can empty the graph (S160).
5. **Oracle triples still show the task is solvable** (90% in the first experiment) when the fact list is short, complete, and in English.

### Limitations

- n=20, one local 8B model, binary QA only.
- Answering context is URI triples from the restored pipeline, not label-rendered GSS output from the paper figures.
- No Recall@k / MRR vs CoLoTa `kg_triples` (GSS-only run; live neighborhood is not the 37-document pool).
- Wikidata 504s and NER leakage are part of the measured system, not filtered out.
- First SPARQL process died after the last cache write; metrics come from a cache-resume answering pass. `gss@5` agreed with the live pass on the 19 printed items.

---

## 5. How to reproduce

From the repo root, with Ollama serving `llama3.1:8b`:

```text
python src/colota_demo/run_suite.py --approaches gss --experiment gss
```

Hand-picked IDs: `src/colota_demo/utils/dataset.py` (`DEMO_IDS`). Mixing weight: `config.json` → `gss.a = 0.75`. Subgraph cache: `out/colota_demo/gss/gss_cache/{id}_a0.75.json` (skip SPARQL on rerun).

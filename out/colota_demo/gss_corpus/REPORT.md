# CoLoTa demo — GSS-only on the 37-triple corpus (n=20)

**Date:** 2026-10-06  
**Live code:** `src/colota_demo/` (`approaches/gss/`, `run_suite.py`; `gss.kg = corpus`)  
**Raw outputs:** `results.csv`, `results.json`, `metrics.json`, `gss_cache/` in this folder  
**Previous GSS (live Wikidata):** [`../gss/REPORT.md`](../gss/REPORT.md)  
**Sister archive (LLM / gold triples / MiniLM RAG):** [`../first_experiment/REPORT.md`](../first_experiment/REPORT.md)  
**Follow-up (iSummary official JAR, Wikidata log):** [`../isummary/REPORT.md`](../isummary/REPORT.md)

Same 20 hand-picked CoLoTa QA items. GSS-only. Initial KG is the **pooled 37 gold triples**, not live Wikidata. NER → 1-hop → prune/rank is unchanged; lookup/fetch/degree use that local graph. Empty subgraph when NER matches nothing.

---

## 1. Motivation

The Wikidata GSS run (`../gss/`) scored **50%** at k=5, tying LLM-only. NER leakage sent SPARQL to the wrong QIDs, and answering saw URI triples.

This run asks the same answering and ranking questions on the **same 37-triple pool as MiniLM RAG**, so Recall@k / MRR / Hits@k are comparable to the first experiment’s retriever.

---

## 2. Setup

- **LLM:** local Ollama `llama3.1:8b`, temperature 0
- **KG:** pooled unique `kg_triples` from the 20 items (37 documents)
- **GSS:** NER (same llama) → match names to local nodes → 1-hop in the 37-graph → prune → `final_score = 0.75 × importance + 0.25 × similarity`
- **Answering:** top-k GSS triples; JSON `TRUE` / `FALSE` / `INSUFFICIENT`
- **If NER misses:** empty subgraph (no question-word fallback)
- **Depths:** k ∈ {1, 3, 5, 10}; primary k = 5
- **Ranking gold:** that item’s `kg_triples` (exact string match), same helpers as RAG  
  Hits@k = first relevant rank ≤ k (`gss_first_relevant_at_k`)

The Cursor-attached shells were aborted mid-run; the Python process kept going and wrote results at 12:00.

---

## 3. Results

### 3.1 Main conditions (k=5)

| Condition | Accuracy | Role |
|-----------|----------|------|
| LLM-only (first experiment) | **50.0%** | parametric baseline |
| Gold triples (first experiment) | **90.0%** | reasoning ceiling |
| RAG@5 (first experiment) | **70.0%** | MiniLM over the 37 triples |
| GSS@5 live Wikidata | **50.0%** | previous GSS run |
| **GSS@5 corpus (this run)** | **40.0%** | GSS ranking over the same 37 triples |
| iSummary@5 (Wikidata log) | **10.0%** | official JAR; see [`../isummary/REPORT.md`](../isummary/REPORT.md) |

Corpus GSS is below LLM-only and well below RAG@5. iSummary on this slice is mostly empty log match / original size-gate abort.

### 3.2 Depth (answering and retrieval)

| k | GSS accuracy | GSS Recall | Hits@k (first gold ≤ k) |
|---|--------------|------------|-------------------------|
| 1 | 35.0% | 25.4% | 45% |
| 3 | 40.0% | 38.3% | 45% |
| 5 | **40.0%** | **38.3%** | **45%** |
| 10 | 40.0% | 38.3% | 45% |

k=3 already saturates retrieval. Rankings are short (often 0–2 triples), so k=5 and k=10 do not add facts.

### 3.3 Ranking vs MiniLM RAG (first experiment)

| Metric | GSS corpus | MiniLM RAG |
|--------|------------|------------|
| MRR (first relevant) | **0.450** | 1.000 |
| Mean RR (all gold triples) | 0.319 | 0.802 |
| Mean first-relevant rank (when any gold is found) | 1.00 | 1.00 |
| Hits@1 / @3 / @5 | 45% / 45% / 45% | 100% / 100% / 100% |
| Recall@5 | 38.3% | 97.5% |

When GSS finds a gold triple, it is at rank 1. It finds **no** gold triple on 11/20 items (Hits = 45%). RAG always put a gold fact first.

### 3.4 Per-item snapshot

See `results.csv`. `INSUFFICIENT` counts as incorrect.

| ID | Gold | GSS@5 | R@5 | Notes |
|----|------|-------|-----|-------|
| S1 | True | False | 100% | Both population triples retrieved; numeric step still fails (same as gold-triples/RAG) |
| S2 | False | INSUFFICIENT | 0% | NER leak; empty subgraph |
| S3 | True | False | 50% | Partial gold |
| S5 | True | INSUFFICIENT | 0% | Empty |
| S13 | False | False | 100% | |
| S37 | False | False | 50% | Multi-hop gold set incomplete |
| S51 | True | INSUFFICIENT | 0% | Empty |
| S87 | False | False | 100% | |
| S95 | False | False | 0% | Empty graph; model said False |
| S131 | False | False | 100% | |
| S142 | True | INSUFFICIENT | 0% | Empty |
| S160 | True | INSUFFICIENT | 0% | Empty |
| S164 | True | INSUFFICIENT | 0% | Empty |
| S175 | False | False | 100% | |
| S184 | False | INSUFFICIENT | 0% | Empty |
| S4 | False | INSUFFICIENT | 0% | Empty |
| S10 | False | False | 0% | Empty graph; model said False |
| S34 | True | False | 67% | |
| S55 | True | INSUFFICIENT | 0% | Empty |
| S124 | True | **True** | 100% | Only True item GSS got right at k=5 |

**Correct at k=5:** S13, S37, S87, S95, S131, S175, S10, S124 (8/20).  
**True items correct at k=5:** S124 only (1/10).  
**INSUFFICIENT at k=5:** 9 items (S2, S5, S51, S142, S160, S164, S184, S4, S55).

Nine non-empty subgraphs (S1, S3, S13, S37, S87, S131, S175, S34, S124) are exactly the Hits@k = 45% set.

---

## 4. Findings

1. **Same 37-triple KG does not make GSS match RAG.** RAG@5 was 70% with Recall@5 97.5%. GSS@5 is 40% with Recall@5 38%.
2. **The bottleneck is still NER leakage, not ranking math.** Leaked QIDs (Q30, Q11424, GMT, …) are not nodes in `(Horsens, population, …)`, so fetch returns nothing. 11/20 summaries are empty; 9 of those abstain.
3. **When NER hits real names, GSS often recovers the gold triples** (R@5 = 100% on S1, S13, S87, S131, S175, S124) and ranks the first gold at 1. Answering can still fail (S1 numeric comparison).
4. **Accuracy is pulled down by True items.** 1/10 True correct vs many Gold-False + empty → False/`INSUFFICIENT`.
5. **Bigger k does nothing here** because the local 1-hop graph is tiny. Contrast the Wikidata run, where extra triples actively hurt (55% → 30% from k=1 to k=10).

### Limitations

- n=20, one local 8B model.
- Empty-on-NER-miss was an explicit choice; a question-word seed would change Hits/Recall.
- Failure tags vs LLM/gold/RAG are empty (GSS-only).
- Attached monitor shells were aborted; metrics come from the surviving Python process that finished at 12:00.

---

## 5. How to reproduce

From the repo root, with Ollama serving `llama3.1:8b`:

```text
python src/colota_demo/run_suite.py --approaches gss --experiment gss_corpus
```

`config.json` → `gss.kg = corpus`, `gss.a = 0.75`. Caches: `gss_cache/{id}_a0.75_corpus.json`. Wikidata SPARQL: `gss.kg = wikidata`.

# CoLoTa demo — toy extract_embed with GSS llama NER (n=20)

**Date:** 2026-10-07  
**Code:** `src/colota_demo/approaches/extract_embed/` + `utils/gss_ner.py`  
**Raw outputs:** `out/colota_demo/extract_embed_ner/` (`results.csv`, `results.json`, `metrics.json`)  
**Shared NER cache:** `out/colota_demo/gss_ner_cache/{id}.json` (identical seeds to PPR-NER)  
**Oracle-linking sister run:** [`../ppr_extract_embed/REPORT_extract_embed.md`](../ppr_extract_embed/REPORT_extract_embed.md)

MiniLM cosine on the entity-incident subset is unchanged. Only the **filter keys** change: GSS NER resource names instead of gold `kg_entities`.

---

## How the method works in theory

KG-RAG often **links** then **embeds**. Gold `kg_entities` made the first stage perfect, so the 75% @k=5 sister run measured cosine on a clean incident subset (with a 86.7% recall ceiling from dropped 2-hop facts).

GSS NER is a noisy linker: the few-shot Wikidata prompt copies Q-ids from examples (Nolan `Q2513`, film `Q11424`, GMT, …). Those strings do not match English `(s, p, o)` nodes in the 37-triple pool, so the incident subset is **empty** and cosine never runs. This run asks whether extract_embed still beats unfiltered MiniLM RAG once it shares GSS’s linker with PPR.

---

## How this algorithm works

1. **NER.** Same cached `extract_entities()` call as PPR-NER (resource names only).
2. **Filter.** Keep a corpus triple iff subject or object matches a resource name (same `name_matches` as gold). Empty subset → empty ranking → `INSUFFICIENT`.
3. **Embed.** Unchanged MiniLM cosine on the kept rows only; top-k → answering prompt.

Gold triples the filter dropped (no matching NER name, or 2-hop) count as recall misses.

---

## Results

### Main conditions (k=5)

| Condition | Accuracy@5 | Seeds |
|-----------|------------|-------|
| extract_embed@5 gold `kg_entities` | **75.0%** | oracle labels |
| MiniLM RAG@5 | 70.0% | no filter |
| PPR@5 GSS NER (same cache) | **40.0%** | GSS NER resources |
| GSS@5 corpus | 40.0% | same llama NER + GSS rank |
| **extract_embed@5 GSS NER (this run)** | **35.0%** | GSS NER resources |
| LLM-only | 50.0% | — |

The filter+cosine toy falls **below** both PPR-NER and GSS corpus, and well below RAG. Oracle linking was doing the work.

### Depth

| k | Recall | extract_embed-NER accuracy | Hits@k |
|---|--------|----------------------------|--------|
| 1 | 25.4% | 35.0% | 45% |
| 3 | 35.8% | 35.0% | 45% |
| 5 | 35.8% | 35.0% | 45% |
| 10 | 35.8% | 35.0% | 45% |

MRR = **0.450**, Mean RR = 0.306. Hits@5 = 45% — **the same 9/20 items** as PPR-NER find a gold triple at rank 1; the other 11 never do. Recall saturates at 35.8% by k=3 (PPR-NER reached 40.0%): the extra ~4% is 2-hop gold PPR can walk to and the filter cannot.

Accuracy is **flat in k**. Empty lists stay empty; the non-empty subsets are already tiny.

### Contrast with PPR-NER (same seeds)

| ID | extract_embed R@5 | PPR R@5 | extract_embed@5 | PPR@5 |
|----|-------------------|---------|-----------------|-------|
| S13 | 50% | 50% | **True** (wrong) | False |
| S37 | 50% | 100% | False | **True** (wrong) |
| S175 | 100% | 100% | **True** (wrong) | False |
| S34 | 67% | 100% | False | False |

S37 is the 2-hop pattern again: PPR Recall@5 = 100%, extract_embed 50%. Answering luck then diverges (extract_embed is correct on S37, PPR is not). Net: PPR-NER 40% vs extract_embed-NER 35%.

### Per-item (k=5)

| ID | Gold | seeds | extract_embed@5 | R@5 |
|----|------|-------|-----------------|-----|
| S1 | True | 3 | False | 100% |
| S2 | False | 2 | INS | 0% |
| S3 | True | 1 | INS | 0% |
| S5 | True | 5 | True | 50% |
| S13 | False | 2 | True | 50% |
| S37 | False | 6 | False | 50% |
| S51 | True | 3 | INS | 0% |
| S87 | False | 2 | False | 100% |
| S95 | False | 4 | False | 0% |
| S131 | False | 2 | False | 100% |
| S142 | True | 2 | INS | 0% |
| S160 | True | 1 | INS | 0% |
| S164 | True | 2 | INS | 0% |
| S175 | False | 2 | True | 100% |
| S184 | False | 1 | INS | 0% |
| S4 | False | 2 | INS | 0% |
| S10 | False | 2 | False | 0% |
| S34 | True | 3 | False | 67% |
| S55 | True | 2 | INS | 0% |
| S124 | True | 2 | True | 100% |

**Correct at k=5:** S5, S37, S87, S95, S131, S10, S124 (7/20).  
**True items correct:** S5, S124 (2/10).  
**INSUFFICIENT at k=5:** the same 9 items as PPR-NER.

---

## Findings

1. **Same NER cache ⇒ same empty set.** Nine `INSUFFICIENT` rows are identical to PPR-NER. Cosine cannot recover a Q-id that is not a node name.
2. **35% < 40% PPR-NER < 70% RAG.** Unfiltered MiniLM does not need NER; the hard gate hurts more than it helps once seeds are GSS-quality.
3. **Gold-seed 75% is not a property of the ranker.** It is a property of `kg_entities`.
4. **Not G-Retriever.** Still MiniLM on a filtered row subset.

```text
python src/colota_demo/run_suite.py --approaches extract_embed --experiment extract_embed_ner --seeds gss_ner
```

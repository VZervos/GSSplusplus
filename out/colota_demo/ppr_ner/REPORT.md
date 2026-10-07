# CoLoTa demo — toy PPR with GSS llama NER (n=20)

**Date:** 2026-10-07  
**Code:** `src/colota_demo/approaches/ppr/` + `utils/gss_ner.py`  
**Raw outputs:** `out/colota_demo/ppr_ner/` (`results.csv`, `results.json`, `metrics.json`)  
**Shared NER cache:** `out/colota_demo/gss_ner_cache/{id}.json`  
**Oracle-linking sister run:** [`../ppr_extract_embed/REPORT_ppr.md`](../ppr_extract_embed/REPORT_ppr.md)

PageRank math is unchanged (undirected s–o graph, α=0.85, triple score = mean of endpoint PPR). Only the **seed list** changes: GSS `extract_entities()` resource names instead of gold `kg_entities`.

---

## How the method works in theory

**Personalized PageRank** scores nodes by a random walk that teleports back to **query seeds**. With gold CoLoTa labels those seeds sit on the right graph nodes, so the walk is an upper bound on ranking-given-perfect-linking (75% @k=5 in the sister run).

A deployed system does not get gold labels. GSS NER is the linker already used by the GSS approach: llama3.1:8b with the original Wikidata few-shot prompt (Q30, Q11424, GMT, Christopher Nolan, …). If that linker emits QIDs that are not English node names in the 37-triple pool, PPR has **no seeds on the graph**, the ranking is empty, and answering must abstain. This run asks whether PPR’s 75% survives once linking is the same noisy NER as GSS.

---

## How this algorithm works

1. **NER (exact GSS).** `approaches.gss.utils.parser.extract_entities(question)` — unmodified few-shot prompt, temp 0, same local `llama3.1:8b`. Cached once per item so extract_embed reuses the identical names.
2. **Seeds.** Entity objects with `type=resource` only (skip `P36`-style properties). Names are matched onto graph nodes with the same normalizer as gold labels (lowercase, `_` → space). `Q11424` does not equal `Horsens`.
3. **Empty match.** No graph node hits any resource name → empty ranking → `INSUFFICIENT`, recall 0.
4. **Rank.** Unchanged NetworkX PageRank α=0.85; triple score \(\frac{1}{2}(\mathrm{PPR}(s)+\mathrm{PPR}(o))\); top-k → answering prompt.

**Not changed:** graph construction, damping, scoring, answering prompt, `DEMO_IDS`, 37-triple pool.

---

## Results

### Main conditions (k=5)

| Condition | Accuracy@5 | Seeds |
|-----------|------------|-------|
| PPR@5 gold `kg_entities` | **75.0%** | oracle labels |
| MiniLM RAG@5 | 70.0% | none (question embedding) |
| GSS@5 corpus | **40.0%** | same llama NER + 1-hop/rank |
| **PPR@5 GSS NER (this run)** | **40.0%** | GSS NER resource names |
| LLM-only | 50.0% | — |

With GSS NER, PPR **ties corpus GSS** and falls **below LLM-only**. The 35-point drop from gold-seed PPR is linking, not PageRank.

### Depth

| k | Recall | PPR-NER accuracy | Hits@k |
|---|--------|------------------|--------|
| 1 | 25.4% | 40.0% | 45% |
| 3 | 38.8% | 40.0% | 45% |
| 5 | 40.0% | 40.0% | 45% |
| 10 | 40.0% | 35.0% | 45% |

MRR = **0.450**, Mean RR = 0.320, mean first-relevant rank = 1.00 (when any gold is found). Hits stay 45% at every k: **11/20 items never retrieve a gold triple**. Retrieval saturates by k=3–5. k=10 is slightly worse for answering.

These retrieval numbers match the shape of corpus GSS (MRR 0.450, Hits 45%, Recall@5 38.3%).

### Per-item (k=5)

| ID | Gold | seeds | PPR@5 | R@5 | Notes |
|----|------|-------|-------|-----|-------|
| S1 | True | 3 | False | 100% | Horsens + Ikast extracted; numeric step still fails |
| S2 | False | 2 | INS | 0% | NER = `Q2513` (Nolan few-shot) + `Q11424`; no Yui |
| S3 | True | 1 | INS | 0% | Resource name missed the 37-graph |
| S5 | True | 5 | True | 50% | `Ulf_Kristersson` survived alongside GMT-board-game leaks |
| S13 | False | 2 | False | 50% | |
| S37 | False | 6 | True | 100% | Full gold ranked; answering flips |
| S51 | True | 3 | INS | 0% | |
| S87 | False | 2 | False | 100% | |
| S95 | False | 4 | False | 0% | Empty graph; False guess (not INS) |
| S131 | False | 2 | False | 100% | |
| S142 | True | 2 | INS | 0% | |
| S160 | True | 1 | INS | 0% | |
| S164 | True | 2 | INS | 0% | |
| S175 | False | 2 | False | 100% | |
| S184 | False | 1 | INS | 0% | |
| S4 | False | 2 | INS | 0% | |
| S10 | False | 2 | False | 0% | Empty graph; False guess |
| S34 | True | 3 | False | 100% | Oracle-reasoning item |
| S55 | True | 2 | INS | 0% | |
| S124 | True | 2 | True | 100% | |

**Correct at k=5:** S5, S13, S87, S95, S131, S175, S10, S124 (8/20).  
**True items correct:** S5, S124 (2/10).  
**INSUFFICIENT at k=5:** 9 items.

S2 is the leakage exhibit: the prompt’s Christopher Nolan / film examples overwrite the actual pet question.

---

## Findings

1. **PPR@5 with GSS NER = GSS corpus @5 = 40%.** Same linker, same 37-graph, different ranker → same answering rate. The gold-seed 75% was oracle linking.
2. **Hits@5 = 45% = GSS corpus.** When NER hits an English name that exists in the pool, PPR still puts a gold triple at rank 1. When it emits only Q-ids, both systems retrieve nothing.
3. **This NER pass is not the GSS-corpus NER pass.** S5 is empty in the GSS-corpus archive and non-empty here (`Ulf_Kristersson`). Fair PPR vs extract_embed comparison uses `gss_ner_cache/`; vs GSS-corpus is the same *prompt*, not the same tokens.
4. **Not HippoRAG.** Still a 37-node undirected walk.

```text
python src/colota_demo/run_suite.py --approaches ppr --experiment ppr_ner --seeds gss_ner
```

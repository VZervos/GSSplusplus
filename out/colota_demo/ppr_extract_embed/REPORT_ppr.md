# CoLoTa demo — toy Personalized PageRank (n=20)

**Date:** 2026-10-06  
**Code:** `src/colota_demo/approaches/ppr/`  
**Raw outputs (this run, with extract_embed):** `out/colota_demo/ppr_extract_embed/` (`results.csv`, `results.json`, `metrics.json`)  
**This is not HippoRAG.** It is a small PageRank ranker on the same 37-triple pool as MiniLM RAG.

---

## How the method works in theory

**Personalized PageRank (PPR)** (Haveliwala, 2003) scores nodes by a biased random walk. Ordinary PageRank asks: if a walker moves along edges at random, where does it spend time in the long run? That is *global* importance (hubs, high degree).

PPR adds a **personalization vector**: with probability \(1-\alpha\) the walker **teleports back to a set of seed nodes** instead of jumping uniformly. The stationary mass is then “importance as seen from the seeds.” In KGQA this is the usual heuristic for a query-focused subgraph (GRAFT-Net, PullNet, HippoRAG): seed on the question’s entities, walk the graph, keep nearby nodes.

On a tiny closed graph the walk cannot invent facts. It can only **re-rank triples that already exist**. Seeds that sit on a gold entity will push mass onto neighbors (and neighbors of neighbors), so a supporting triple that does *not* mention the seed by name can still rise (e.g. pet → breed). Isolated components that do not contain a seed stay near score 0.

\(\alpha = 0.85\) is the usual damping: 85% follow an edge, 15% jump to a seed.

---

## How this algorithm works

**Graph.** The 20-item CoLoTa pool is parsed as `(s, p, o)` strings. Nodes are **subjects and objects only**. Each triple becomes an **undirected** edge `s — o`. Predicates are not nodes; they do not participate in the walk.

**Seeds.** Gold CoLoTa `kg_entities` **labels** (not QIDs, not llama NER). A graph node matches a seed if the names are equal after lowercasing and `_` → space. No Falcon, no question embedding.

**If no seed matches any node:** empty ranking → answering `INSUFFICIENT`, recall 0.

**Scores.** NetworkX `pagerank` with equal personalization `1/|seeds|` on the matched nodes (dangling mass uses the same vector). Triple score = \(\frac{1}{2}(\mathrm{PPR}(s)+\mathrm{PPR}(o))\). All 37 triples are sorted by that score (zeros last, tie-break by string). Top-k go to llama3.1:8b (`TRUE` / `FALSE` / `INSUFFICIENT`).

**Metrics.** Same as RAG: accuracy@k (`INSUFFICIENT` = wrong), Recall@k / MRR / Mean RR / Hits@k vs that item’s English `kg_triples`.

**Protocol.** Same `DEMO_IDS` (10 True / 10 False), k ∈ {1,3,5,10}, primary k=5, local Ollama temperature 0.

---

## Results

### Main conditions (k=5)

| Condition | Accuracy@5 | Role |
|-----------|------------|------|
| LLM-only (first experiment) | 50.0% | parametric |
| Gold triples (first experiment) | 90.0% | oracle facts |
| MiniLM RAG@5 (first experiment) | 70.0% | cosine on 37 triples |
| GSS@5 corpus | 40.0% | NER + 1-hop on 37 triples |
| **PPR@5 (this run)** | **75.0%** | gold-entity seeds + walk |

PPR@5 matches extract_embed@5 (75%) and is **above RAG@5 (70%)** on answering, with almost the same retrieval numbers as RAG.

### Depth

| k | Recall | PPR accuracy | Hits@k |
|---|--------|--------------|--------|
| 1 | 64.6% | 70.0% | 100% |
| 3 | 96.2% | **85.0%** | 100% |
| 5 | 97.5% | 75.0% | 100% |
| 10 | 97.5% | 80.0% | 100% |

MRR = **1.000**, Mean RR = 0.800, mean first-relevant rank = 1.00. Every item has a gold triple at rank 1. Best **answering** depth is **k=3 (85%)**, not k=5: extra triples add distractors (same pattern as RAG).

Recall@10 stays 97.5% (not 100%): at least one gold string is never ranked in the top 10 for some item (S5 stays at R=50% at every k — the second gold fact is not pulled up enough, or is a disconnected/low-mass node).

### Per-item (k=5)

| ID | Gold | PPR@5 | R@5 | Notes |
|----|------|-------|-----|-------|
| S1 | True | False | 100% | Both populations retrieved; numeric comparison still fails (same as gold-triples/RAG) |
| S2 | False | **True** | 100% | Walk reaches dachshund; model still says True |
| S3 | True | True | 100% | |
| S5 | True | True | 50% | Only one of two gold facts in top-5 |
| S13 | False | False | 100% | |
| S37 | False | **True** | 100% | Full gold set; answering flips |
| S51 | True | True | 100% | |
| S87 | False | False | 100% | |
| S95 | False | False | 100% | |
| S131 | False | False | 100% | |
| S142 | True | True | 100% | |
| S160 | True | True | 100% | |
| S164 | True | True | 100% | |
| S175 | False | False | 100% | |
| S184 | False | False | 100% | |
| S4 | False | **True** | 100% | Wrong at k=5, correct at k=3 and k=10 |
| S10 | False | False | 100% | Wrong at k=1, then correct |
| S34 | True | False | 100% | Genre matching; also fails gold-triples in the first experiment |
| S55 | True | True | 100% | |
| S124 | True | True | 100% | |

**Wrong at k=5:** S1, S2, S37, S4, S34 (5/20).  
**True items correct at k=5:** 8/10 (misses S1, S34).  
No empty rankings: every question’s `kg_entities` hit the 37-graph.

S2 is the graph-walk illustration: extract_embed never sees `(Sablé, animal breed, dachshund)` because Sablé is not a listed entity; PPR does, and R@5=100%. Answering still fails.

---

## Findings

1. **Oracle entity linking + PPR on this pool is a strong retriever** (MRR 1, Recall@5 97.5%), essentially MiniLM RAG’s retrieval with a different score.
2. **Answering peaks at k=3 (85%)**, better than RAG@3 (75% in the first experiment) on this run. Treat that gap cautiously: the two answering passes are different days/processes.
3. **Seeds are gold `kg_entities`.** This isolates ranking from GSS NER leakage. It is an upper bound on “PPR if linking is perfect,” not a deployed linker.
4. **Not HippoRAG:** no OpenIE, no document index, no passage nodes.

```text
python src/colota_demo/run_suite.py --approaches ppr --experiment ppr
```

`config.json` → `ppr.alpha = 0.85`.

# CoLoTa demo — toy entity filter + MiniLM (n=20)

**Date:** 2026-10-06  
**Code:** `src/colota_demo/approaches/extract_embed/`  
**Raw outputs (this run, with PPR):** `out/colota_demo/ppr_extract_embed/` (`results.csv`, `results.json`, `metrics.json`)  
**This is not G-Retriever or HippoRAG.** It is MiniLM RAG with a hard gate on gold entity names.

---

## How the method works in theory

KG-RAG often has two stages:

1. **Extraction / linking** — decide which KG nodes the question is about (topic entities).
2. **Semantic embeddings** — score remaining triples by cosine similarity between the question text and the triple text.

The idea is that the full KG (or even a 37-triple pool) contains distractors from *other* questions. If you first keep only triples incident to the topic entities, the embedding ranker searches a smaller, more relevant set. That is the lightweight form of “entity-filtered retrieval” used in many KG-RAG pipelines (link, then retrieve around those nodes). It is **not** G-Retriever: there is no Prize-Collecting Steiner Tree, no connectivity constraint, and no prize on every node in the graph.

The cost of the gate: any gold fact whose subject and object are **not** a listed topic entity is dropped forever (typical 2-hop evidence: pet name listed, breed triple only mentions the pet). Embeddings cannot recover what the filter deleted.

---

## How this algorithm works

**Corpus.** Same 37 unique gold triples as RAG.

**Extraction (oracle).** Gold CoLoTa `kg_entities` **labels**, same name match as PPR (case-insensitive, `_` → space). No llama NER, no Falcon, no ReFinED.

**Filter.** Keep a corpus triple iff its **subject or object** equals a seed name. Predicates are not matched. Empty subset → empty ranking → `INSUFFICIENT`.

**Embeddings.** `sentence-transformers/all-MiniLM-L6-v2` (same as RAG). The full corpus is encoded once. At query time the question is encoded and cosine is computed **only against the kept rows**. Sort that subset; take top-k.

**Answering.** Same llama3.1:8b JSON `TRUE` / `FALSE` / `INSUFFICIENT` protocol as RAG/GSS. Metrics: accuracy@k, Recall@k, MRR, Mean RR, Hits@k vs that item’s `kg_triples`. Ranks are 1-indexed **inside the filtered list**; gold triples that the filter dropped count as misses.

**Protocol.** Same `DEMO_IDS`, k ∈ {1,3,5,10}, primary k=5.

---

## Results

### Main conditions (k=5)

| Condition | Accuracy@5 | Role |
|-----------|------------|------|
| LLM-only (first experiment) | 50.0% | parametric |
| Gold triples (first experiment) | 90.0% | oracle facts |
| MiniLM RAG@5 (first experiment) | 70.0% | cosine on **all** 37 triples |
| **extract_embed@5 (this run)** | **75.0%** | cosine on **entity-incident** triples |
| PPR@5 (same process) | 75.0% | walk from the same seeds |

Same answering accuracy as toy PPR at k=5; **weaker coverage** than RAG/PPR because of the gate.

### Depth

| k | Recall | extract_embed accuracy | Hits@k |
|---|--------|------------------------|--------|
| 1 | 64.6% | 75.0% | 100% |
| 3 | 86.7% | **80.0%** | 100% |
| 5 | 86.7% | 75.0% | 100% |
| 10 | 86.7% | 75.0% | 100% |

MRR = **1.000**, Mean RR = 0.753, mean first-relevant rank = 1.00. Hits@1 = 100%: every item still has **some** gold triple that mentions a listed entity, so the first rank in the *subset* is always gold. Recall saturates at **86.7%** by k=3 — the missing ~13% is gold facts the filter removed, not ranking depth. k=10 cannot help: those triples are gone.

Best answering depth is **k=3 (80%)**.

### Contrast with PPR on the same items

The filter vs walk shows up where gold evidence is 2-hop:

| ID | Gold | extract_embed R@5 | PPR R@5 | extract_embed@5 | PPR@5 |
|----|------|-------------------|---------|-----------------|-------|
| S2 | False | **50%** | 100% | True (wrong) | True (wrong) |
| S3 | True | **50%** | 100% | True | True |
| S37 | False | **50%** | 100% | True (wrong) | True (wrong) |
| S34 | True | **33%** | 100% | False | False |
| S5 | True | 50% | 50% | True | True |

S2: only `(Yui Yuigahama, has pet, Sablé)` is incident to the listed entity; the dachshund triple is dropped. PPR still ranks it. Both models answer True anyway.

### Per-item (k=5)

| ID | Gold | extract_embed@5 | R@5 | Notes |
|----|------|-----------------|-----|-------|
| S1 | True | False | 100% | Numeric comparison (same family as gold-triples fail) |
| S2 | False | **True** | 50% | Breed triple filtered out |
| S3 | True | True | 50% | Partial gold; still correct |
| S5 | True | True | 50% | |
| S13 | False | False | 100% | Wrong at k=1, then correct |
| S37 | False | **True** | 50% | Continents / extra hops dropped |
| S51 | True | True | 100% | |
| S87 | False | False | 100% | |
| S95 | False | False | 100% | |
| S131 | False | False | 100% | |
| S142 | True | True | 100% | |
| S160 | True | True | 100% | |
| S164 | True | True | 100% | |
| S175 | False | **True** | 100% | Full gold in subset; answering fails |
| S184 | False | False | 100% | |
| S4 | False | False | 100% | |
| S10 | False | False | 100% | |
| S34 | True | False | 33% | Only one of three gold triples incident to listed names |
| S55 | True | True | 100% | |
| S124 | True | True | 100% | |

**Wrong at k=5:** S1, S2, S37, S175, S34 (5/20).  
**True items correct at k=5:** 8/10 (misses S1, S34).  
No empty rankings.

---

## Findings

1. **Oracle linking + MiniLM on the incident subset is enough to beat RAG@5 answering (75% vs 70%)** on this slice, even though Recall@5 is **worse** (86.7% vs 97.5%). Fewer distractors from *other* questions can outweigh missing 2-hop gold.
2. **The gate is the retrieval ceiling.** Recall never exceeds 86.7% no matter how large k is.
3. **k=3 is the best answering depth (80%)**; k=5/10 add nothing in recall and can hurt accuracy.
4. **Not a published system.** Same seeds as toy PPR; the difference is filter+cosine vs graph walk.

```text
python src/colota_demo/run_suite.py --approaches extract_embed --experiment extract_embed
```
